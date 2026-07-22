function _byte_vocab()
    return Dict(identifier => UInt8[identifier] for identifier in 0:255)
end

function _gpt2_byte_decoder()
    byte_values = vcat(
        collect(Int('!'):Int('~')),
        collect(Int('¡'):Int('¬')),
        collect(Int('®'):Int('ÿ')),
    )
    codepoints = copy(byte_values)
    offset = 0
    for byte in 0:255
        byte in byte_values && continue
        push!(byte_values, byte)
        push!(codepoints, 256 + offset)
        offset += 1
    end
    return Dict(Char(codepoint) => UInt8(byte) for (byte, codepoint) in zip(byte_values, codepoints))
end

function _gpt2_token_bytes(token::AbstractString, decoder)
    return UInt8[decoder[character] for character in token]
end

function _load_gpt2_tokenizer(fixture_root)
    decoder = _gpt2_byte_decoder()
    encoded_vocab = JSON.parsefile(joinpath(fixture_root, "gpt2_vocab.json"))
    vocab = Dict(
        Int(identifier) => _gpt2_token_bytes(token, decoder) for
        (token, identifier) in encoded_vocab
    )
    merges = Tuple{Vector{UInt8},Vector{UInt8}}[]
    for line in eachline(joinpath(fixture_root, "gpt2_merges.txt"))
        tokens = split(chomp(line), ' '; keepempty=false)
        length(tokens) == 2 || continue
        push!(
            merges,
            (_gpt2_token_bytes(tokens[1], decoder), _gpt2_token_bytes(tokens[2], decoder)),
        )
    end
    return BPETokenizer(vocab, merges, ["<|endoftext|>"])
end

@testset "BPE tokenizer native behavior" begin
    vocab = _byte_vocab()
    vocab[256] = collect(codeunits("he"))
    vocab[257] = collect(codeunits("hel"))
    vocab[258] = collect(codeunits("hell"))
    vocab[259] = collect(codeunits("hello"))
    merges = [
        (UInt8['h'], UInt8['e']),
        (collect(codeunits("he")), UInt8['l']),
        (collect(codeunits("hel")), UInt8['l']),
        (collect(codeunits("hell")), UInt8['o']),
    ]
    tokenizer = BPETokenizer(vocab, merges)

    @test encode(tokenizer, "") == Int[]
    @test encode(tokenizer, "hello") == [259]
    @test decode(tokenizer, encode(tokenizer, "hello 🙃")) == "hello 🙃"
    @test collect(encode_iterable(tokenizer, ["hello", " ", "world"])) ==
          encode(tokenizer, "hello world")
    @test_throws ArgumentError decode(tokenizer, [10_000])

    invalid_vocab = _byte_vocab()
    invalid_tokenizer = BPETokenizer(invalid_vocab, Tuple{Vector{UInt8},Vector{UInt8}}[])
    @test decode(invalid_tokenizer, [255]) == "�"
end

@testset "BPE tokenizer GPT-2 fixture parity" begin
    fixture_root = normpath(joinpath(@__DIR__, "..", "..", "tests", "fixtures"))
    tokenizer = _load_gpt2_tokenizer(fixture_root)
    cases = [
        "" => Int[],
        "s" => [82],
        "🙃" => [8582, 247, 225],
        "Hello, how are you?" => [15496, 11, 703, 389, 345, 30],
        "Héllò hôw are ü? 🙃" =>
            [39, 2634, 297, 127, 110, 289, 27083, 86, 389, 6184, 120, 30, 12520, 247, 225],
        "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>" => [
            39,
            2634,
            297,
            127,
            110,
            289,
            27083,
            86,
            220,
            50256,
            50256,
            389,
            6184,
            120,
            30,
            12520,
            247,
            225,
            50256,
        ],
    ]
    for (text, expected) in cases
        identifiers = encode(tokenizer, text)
        @test identifiers == expected
        @test decode(tokenizer, identifiers) == text
    end

    overlapping = BPETokenizer(
        tokenizer.vocab,
        Tuple{Vector{UInt8},Vector{UInt8}}[],
        ["<|endoftext|>", "<|endoftext|><|endoftext|>"],
    )
    text = "x<|endoftext|><|endoftext|>y<|endoftext|>"
    identifiers = encode(overlapping, text)
    double_identifier = overlapping.user_special_tokens["<|endoftext|><|endoftext|>"]
    single_identifier = overlapping.user_special_tokens["<|endoftext|>"]
    @test count(==(double_identifier), identifiers) == 1
    @test count(==(single_identifier), identifiers) == 1
    @test decode(overlapping, identifiers) == text
    @test_throws ArgumentError encode(
        overlapping,
        text;
        allowed_special=Set{String}(),
        disallowed_special=:all,
    )

    for filename in ("address.txt", "german.txt", "tinystories_sample.txt")
        text = read(joinpath(fixture_root, filename), String)
        identifiers = encode(tokenizer, text)
        @test decode(tokenizer, identifiers) == text
    end

    input = eachline(joinpath(fixture_root, "tinystories_sample.txt"); keep=true)
    streamed = collect(encode_iterable(tokenizer, input))
    @test decode(tokenizer, streamed) == read(
        joinpath(fixture_root, "tinystories_sample.txt"),
        String,
    )
end

function _hex_merges(merges)
    return [(bytes2hex(left), bytes2hex(right)) for (left, right) in merges]
end

function _contains_special_prefix(bytes)
    length(bytes) < 2 && return false
    return any(
        bytes[index] == UInt8('<') && bytes[index + 1] == UInt8('|') for
        index in 1:(length(bytes) - 1)
    )
end

@testset "tiny BPE trainer Python parity" begin
    cases = [
        (
            "hello hello<|endoftext|>world",
            258,
            [("6c", "6f")],
            Dict(257 => "6c6f"),
        ),
        ("ab ac", 259, [("61", "63"), ("61", "62")], Dict(257 => "6163", 258 => "6162")),
        ("aa<|endoftext|>aa", 259, [("61", "61")], Dict(257 => "6161")),
        (
            "low low lower",
            260,
            [("6f", "77"), ("6c", "6f77"), ("20", "6c6f77")],
            Dict(257 => "6f77", 258 => "6c6f77", 259 => "206c6f77"),
        ),
    ]

    mktempdir() do directory
        path = joinpath(directory, "corpus.txt")
        for (corpus, vocab_size, expected_merges, expected_tail) in cases
            write(path, corpus)
            serial_vocab, serial_merges = train_bpe(
                path,
                vocab_size,
                ["<|endoftext|>"];
                parallel=false,
                report_progress=false,
            )
            parallel_vocab, parallel_merges = train_bpe(
                path,
                vocab_size,
                ["<|endoftext|>"];
                parallel=true,
                report_progress=false,
            )
            @test _hex_merges(serial_merges) == expected_merges
            @test Dict(identifier => bytes2hex(serial_vocab[identifier]) for identifier in keys(expected_tail)) ==
                  expected_tail
            @test parallel_vocab == serial_vocab
            @test parallel_merges == serial_merges
            @test all(
                !_contains_special_prefix(bytes) for
                (identifier, bytes) in serial_vocab if identifier != 0
            )
        end
    end
end
