const _ByteToken = Tuple{Vararg{UInt8}}
const _BytePair = Tuple{_ByteToken,_ByteToken}
const _GPT2_PRETOKEN_PATTERN =
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"

_byte_token(bytes) = Tuple(UInt8(byte) for byte in bytes)

"""
    BPETokenizer(vocab, merges, special_tokens=String[])

Construct a byte-level BPE tokenizer. External token IDs are zero-based and
`vocab` maps them to byte vectors. `merges` is ordered from highest to lowest
priority. Missing user-defined special tokens are appended after the highest
existing token ID.
"""
struct BPETokenizer
    vocab::Dict{Int,Vector{UInt8}}
    inverse_vocab::Dict{_ByteToken,Int}
    merge_rank::Dict{_BytePair,Int}
    user_special_tokens::Dict{String,Int}
    assumed_special_tokens::Dict{_ByteToken,Int}
    pretoken_cache::Dict{String,Vector{Int}}
end

function BPETokenizer(
    vocab::AbstractDict{<:Integer,<:AbstractVector{UInt8}},
    merges::AbstractVector,
    special_tokens::AbstractVector{<:AbstractString}=String[],
)
    isempty(vocab) && throw(ArgumentError("BPE vocabulary must not be empty"))
    any(isempty, special_tokens) &&
        throw(ArgumentError("BPE special tokens must not be empty strings"))

    owned_vocab = Dict{Int,Vector{UInt8}}(
        Int(identifier) => collect(UInt8, bytes) for (identifier, bytes) in vocab
    )
    inverse_vocab = Dict{_ByteToken,Int}(
        _byte_token(bytes) => identifier for (identifier, bytes) in owned_vocab
    )
    length(inverse_vocab) == length(owned_vocab) ||
        throw(ArgumentError("BPE vocabulary contains duplicate byte tokens"))

    next_identifier = maximum(keys(owned_vocab)) + 1
    user_special = Dict{String,Int}()
    for token_value in special_tokens
        token = String(token_value)
        bytes = _byte_token(codeunits(token))
        identifier = get(inverse_vocab, bytes, next_identifier)
        if identifier == next_identifier
            owned_vocab[identifier] = collect(bytes)
            inverse_vocab[bytes] = identifier
            next_identifier += 1
        end
        user_special[token] = identifier
    end

    merge_rank = Dict{_BytePair,Int}()
    for (rank, merge) in enumerate(merges)
        length(merge) == 2 || throw(ArgumentError("each BPE merge must contain two byte tokens"))
        pair = (_byte_token(merge[1]), _byte_token(merge[2]))
        get!(merge_rank, pair, rank)
    end

    assumed_special = Dict{_ByteToken,Int}()
    nul_identifier = get(inverse_vocab, (UInt8(0),), -1)
    if nul_identifier > 0
        for identifier in 0:(nul_identifier - 1)
            haskey(owned_vocab, identifier) || continue
            bytes = _byte_token(owned_vocab[identifier])
            assumed_special[bytes] = identifier
        end
    end

    cache = Dict{String,Vector{Int}}()
    for (token, identifier) in user_special
        cache[token] = [identifier]
    end
    return BPETokenizer(
        owned_vocab,
        inverse_vocab,
        merge_rank,
        user_special,
        assumed_special,
        cache,
    )
end

function _encode_pretoken(tokenizer::BPETokenizer, pretoken::AbstractString)
    text = String(pretoken)
    cached = get(tokenizer.pretoken_cache, text, nothing)
    cached === nothing || return cached

    bytes = _byte_token(codeunits(text))
    assumed = get(tokenizer.assumed_special_tokens, bytes, nothing)
    if assumed !== nothing
        identifiers = [assumed]
        tokenizer.pretoken_cache[text] = identifiers
        return identifiers
    end

    tokens = _ByteToken[(byte,) for byte in bytes]
    while length(tokens) > 1
        best_rank = typemax(Int)
        best_index = 0
        for index in 1:(length(tokens) - 1)
            rank = get(tokenizer.merge_rank, (tokens[index], tokens[index + 1]), best_rank)
            if rank < best_rank
                best_rank = rank
                best_index = index
            end
        end
        best_index == 0 && break
        tokens[best_index] = (tokens[best_index]..., tokens[best_index + 1]...)
        deleteat!(tokens, best_index + 1)
    end

    identifiers = Int[]
    sizehint!(identifiers, length(tokens))
    for token in tokens
        identifier = get(tokenizer.inverse_vocab, token, nothing)
        identifier === nothing && throw(
            ArgumentError(
                "pretoken contains byte sequence $(collect(token)) absent from the vocabulary",
            ),
        )
        push!(identifiers, identifier)
    end
    tokenizer.pretoken_cache[text] = identifiers
    return identifiers
end

function _configured_special_strings(tokenizer::BPETokenizer)
    result = collect(keys(tokenizer.user_special_tokens))
    for bytes in keys(tokenizer.assumed_special_tokens)
        token = transcode(String, collect(bytes))
        isvalid(token) && push!(result, token)
    end
    return unique(result)
end

function _special_set(tokenizer::BPETokenizer, selection)
    if selection === :all || selection == "all"
        return Set(keys(tokenizer.user_special_tokens))
    end
    return Set(String(token) for token in selection)
end

function _first_special(text::String, tokens)
    best_token = nothing
    best_range = nothing
    for token in tokens
        isempty(token) && continue
        range = findfirst(token, text)
        range === nothing && continue
        if best_range === nothing || first(range) < first(best_range) ||
           (first(range) == first(best_range) && ncodeunits(token) > ncodeunits(best_token))
            best_token = token
            best_range = range
        end
    end
    return best_token, best_range
end

function _special_fragments(text::String, allowed::Set{String})
    isempty(text) && return Tuple{Bool,String}[]
    isempty(allowed) && return [(false, text)]

    fragments = Tuple{Bool,String}[]
    position = firstindex(text)
    while position <= lastindex(text)
        suffix = SubString(text, position)
        token, relative_range = _first_special(String(suffix), allowed)
        if relative_range === nothing
            push!(fragments, (false, String(suffix)))
            break
        end
        start_index = position + first(relative_range) - 1
        stop_index = position + last(relative_range) - 1
        if start_index > position
            push!(fragments, (false, String(SubString(text, position, prevind(text, start_index)))))
        end
        push!(fragments, (true, token))
        position = nextind(text, stop_index)
    end
    return fragments
end

function _pretokens(text::AbstractString, pretokenization)
    isempty(text) && return String[]
    pretokenization === false && return [String(text)]
    pattern = pretokenization === true ? _GPT2_PRETOKEN_PATTERN : pretokenization
    pattern isa Regex || throw(
        ArgumentError("pretokenization must be true, false, or a Regex"),
    )
    return [String(match.match) for match in eachmatch(pattern, text)]
end

"""
    encode(tokenizer, text; pretokenization=true, allowed_special=:all,
           disallowed_special=String[])

Encode `text` into zero-based token IDs. Longer allowed special tokens take
precedence when configured tokens overlap. `disallowed_special=:all` rejects
every configured special token not explicitly allowed.
"""
function encode(
    tokenizer::BPETokenizer,
    text_value::AbstractString;
    pretokenization=true,
    allowed_special=:all,
    disallowed_special=String[],
)
    text = String(text_value)
    allowed = _special_set(tokenizer, allowed_special)
    disallowed = if disallowed_special === :all || disallowed_special == "all"
        setdiff(Set(_configured_special_strings(tokenizer)), allowed)
    else
        Set(String(token) for token in disallowed_special)
    end
    token, range = _first_special(text, disallowed)
    range === nothing || throw(
        ArgumentError(
            "input at byte range $(first(range)):$(last(range)) contains disallowed special token $(repr(token))",
        ),
    )

    identifiers = Int[]
    for (is_special, fragment) in _special_fragments(text, allowed)
        if is_special && haskey(tokenizer.user_special_tokens, fragment)
            push!(identifiers, tokenizer.user_special_tokens[fragment])
            continue
        end
        bytes = _byte_token(codeunits(fragment))
        if is_special && haskey(tokenizer.assumed_special_tokens, bytes)
            push!(identifiers, tokenizer.assumed_special_tokens[bytes])
            continue
        end
        for pretoken in _pretokens(fragment, pretokenization)
            append!(identifiers, _encode_pretoken(tokenizer, pretoken))
        end
    end
    return identifiers
end

"""
    encode_iterable(tokenizer, iterable; kwargs...)

Lazily encode each string yielded by `iterable`. Chunk boundaries are semantic
boundaries, matching the Python adapter's iterable behavior.
"""
function encode_iterable(tokenizer::BPETokenizer, iterable; kwargs...)
    return Iterators.flatten(
        (encode(tokenizer, String(chunk); kwargs...) for chunk in iterable),
    )
end

"""
    decode(tokenizer, identifiers)

Concatenate vocabulary bytes and decode them as UTF-8. Invalid byte sequences
are replaced during UTF-8 transcoding. Unknown IDs raise `ArgumentError`.
"""
function decode(tokenizer::BPETokenizer, identifiers)
    bytes = UInt8[]
    for identifier_value in identifiers
        identifier = Int(identifier_value)
        haskey(tokenizer.vocab, identifier) ||
            throw(ArgumentError("token ID $identifier is absent from the vocabulary"))
        append!(bytes, tokenizer.vocab[identifier])
    end
    raw = String(bytes)
    isvalid(raw) && return raw
    output = IOBuffer()
    for character in raw
        print(output, isvalid(character) ? character : '\ufffd')
    end
    return String(take!(output))
end

function _count_pretokens(fragment::String, pretokenization)
    counts = Dict{String,Int}()
    for pretoken in _pretokens(fragment, pretokenization)
        counts[pretoken] = get(counts, pretoken, 0) + 1
    end
    return counts
end

function _merge_counts!(destination::Dict{String,Int}, source::Dict{String,Int})
    for (pretoken, count) in source
        destination[pretoken] = get(destination, pretoken, 0) + count
    end
    return destination
end

function _lexless(left::_ByteToken, right::_ByteToken)
    shared = min(length(left), length(right))
    for index in 1:shared
        left[index] == right[index] || return left[index] < right[index]
    end
    return length(left) < length(right)
end

function _pair_lexless(left::Tuple{Int,Int}, right::Tuple{Int,Int}, vocab)
    left_first = vocab[left[1]]
    right_first = vocab[right[1]]
    left_first == right_first || return _lexless(left_first, right_first)
    return _lexless(vocab[left[2]], vocab[right[2]])
end

struct _BPECandidate
    pair::Tuple{Int,Int}
    count::Int
end

function _candidate_precedes(left::_BPECandidate, right::_BPECandidate, vocab)
    left.count == right.count || return left.count > right.count
    return _pair_lexless(right.pair, left.pair, vocab)
end

function _heap_push!(heap::Vector{_BPECandidate}, candidate::_BPECandidate, vocab)
    push!(heap, candidate)
    child = length(heap)
    while child > 1
        parent = child ÷ 2
        _candidate_precedes(heap[child], heap[parent], vocab) || break
        heap[parent], heap[child] = heap[child], heap[parent]
        child = parent
    end
    return heap
end

function _heap_pop!(heap::Vector{_BPECandidate}, vocab)
    isempty(heap) && throw(ArgumentError("cannot pop an empty BPE candidate heap"))
    candidate = heap[1]
    tail = pop!(heap)
    isempty(heap) && return candidate
    heap[1] = tail
    parent = 1
    while true
        left = 2 * parent
        left > length(heap) && break
        right = left + 1
        child = right <= length(heap) &&
                _candidate_precedes(heap[right], heap[left], vocab) ? right : left
        _candidate_precedes(heap[child], heap[parent], vocab) || break
        heap[parent], heap[child] = heap[child], heap[parent]
        parent = child
    end
    return candidate
end

function _sequence_pair_counts(sequence)
    counts = Dict{Tuple{Int,Int},Int}()
    length(sequence) < 2 && return counts
    for index in 1:(length(sequence) - 1)
        pair = (sequence[index], sequence[index + 1])
        counts[pair] = get(counts, pair, 0) + 1
    end
    return counts
end

function _initial_pair_index(sequences)
    counts = Dict{Tuple{Int,Int},Int}()
    sequence_indices = Dict{Tuple{Int,Int},Set{Int}}()
    for (sequence_index, (sequence, frequency)) in enumerate(sequences)
        for (pair, occurrences) in _sequence_pair_counts(sequence)
            counts[pair] = get(counts, pair, 0) + frequency * occurrences
            push!(get!(Set{Int}, sequence_indices, pair), sequence_index)
        end
    end
    return counts, sequence_indices
end

function _next_pair!(heap, authoritative_counts, vocab)
    while !isempty(heap)
        candidate = _heap_pop!(heap, vocab)
        authoritative = get(authoritative_counts, candidate.pair, 0)
        authoritative <= 0 && continue
        candidate.count == authoritative && return candidate.pair
        candidate.count < authoritative &&
            _heap_push!(heap, _BPECandidate(candidate.pair, authoritative), vocab)
    end
    return nothing
end

function _merge_pair(sequence::Vector{Int}, pair::Tuple{Int,Int}, identifier::Int)
    result = Int[]
    sizehint!(result, length(sequence))
    index = 1
    while index <= length(sequence)
        if index < length(sequence) &&
           sequence[index] == pair[1] && sequence[index + 1] == pair[2]
            push!(result, identifier)
            index += 2
        else
            push!(result, sequence[index])
            index += 1
        end
    end
    return result
end

"""
    train_bpe(input_path, vocab_size, special_tokens;
              pretokenization=true, parallel=true, report_progress=true)

Train byte-level BPE and return `(vocab, merges)` using zero-based token IDs.
Special tokens occupy the beginning of the vocabulary, form hard document
boundaries, and do not contribute merge statistics. Equal-frequency pairs use
the lexicographically greatest byte pair, matching the Python implementation.

`parallel=true` parallelizes only independent pretoken counting with task-local
dictionaries. Merge selection remains deterministic and serial. Candidate
selection uses a lazy max-heap validated against authoritative incremental pair
counts; each merge revisits only pretokens containing the selected pair.
"""
function train_bpe(
    input_path::AbstractString,
    vocab_size::Integer,
    special_tokens::AbstractVector{<:AbstractString};
    pretokenization=true,
    parallel::Union{Bool,Integer}=true,
    report_progress::Bool=true,
)
    isempty(special_tokens) && throw(
        ArgumentError("special_tokens must contain at least one boundary token"),
    )
    any(isempty, special_tokens) &&
        throw(ArgumentError("BPE special tokens must not be empty strings"))
    report_progress # accepted for adapter parity; large-corpus progress is deferred

    corpus = read(input_path, String)
    boundary_tokens = Set(String(token) for token in special_tokens)
    fragments = [
        fragment for (is_special, fragment) in _special_fragments(corpus, boundary_tokens) if
        !is_special && !isempty(fragment)
    ]

    pretoken_counts = Dict{String,Int}()
    use_parallel = parallel !== false && Threads.nthreads(:default) > 1 && length(fragments) > 1
    if use_parallel
        tasks = [Threads.@spawn _count_pretokens(fragment, pretokenization) for fragment in fragments]
        for task in tasks
            _merge_counts!(pretoken_counts, fetch(task))
        end
    else
        for fragment in fragments
            _merge_counts!(pretoken_counts, _count_pretokens(fragment, pretokenization))
        end
    end

    vocab_internal = Dict{Int,_ByteToken}()
    for (offset, token) in enumerate(special_tokens)
        vocab_internal[offset - 1] = _byte_token(codeunits(token))
    end
    byte_offset = length(special_tokens)
    for byte in 0:255
        vocab_internal[byte_offset + byte] = (UInt8(byte),)
    end

    sequences = Tuple{Vector{Int},Int}[]
    for pretoken in sort!(collect(keys(pretoken_counts)))
        sequence = [byte_offset + Int(byte) for byte in codeunits(pretoken)]
        push!(sequences, (sequence, pretoken_counts[pretoken]))
    end

    pair_counts, pair_sequences = _initial_pair_index(sequences)
    candidate_heap = _BPECandidate[]
    for (pair, count) in pair_counts
        _heap_push!(candidate_heap, _BPECandidate(pair, count), vocab_internal)
    end

    merges_internal = _BytePair[]
    target_merges = max(0, Int(vocab_size) - length(vocab_internal))
    for _ in 1:target_merges
        pair = _next_pair!(candidate_heap, pair_counts, vocab_internal)
        pair === nothing && break
        identifier = length(vocab_internal)
        merged_token = (vocab_internal[pair[1]]..., vocab_internal[pair[2]]...)
        push!(merges_internal, (vocab_internal[pair[1]], vocab_internal[pair[2]]))
        vocab_internal[identifier] = merged_token

        affected_pairs = Set{Tuple{Int,Int}}()
        affected_sequences = collect(get(pair_sequences, pair, Set{Int}()))
        for sequence_index in affected_sequences
            sequence, frequency = sequences[sequence_index]
            before = _sequence_pair_counts(sequence)
            get(before, pair, 0) == 0 && continue
            merged_sequence = _merge_pair(sequence, pair, identifier)
            after = _sequence_pair_counts(merged_sequence)
            local_pairs = Set{Tuple{Int,Int}}(keys(before))
            union!(local_pairs, keys(after))
            for local_pair in local_pairs
                before_count = get(before, local_pair, 0)
                after_count = get(after, local_pair, 0)
                if before_count != after_count
                    pair_counts[local_pair] =
                        get(pair_counts, local_pair, 0) +
                        frequency * (after_count - before_count)
                    push!(affected_pairs, local_pair)
                end
                indices = get!(Set{Int}, pair_sequences, local_pair)
                if after_count > 0
                    push!(indices, sequence_index)
                else
                    delete!(indices, sequence_index)
                    isempty(indices) && delete!(pair_sequences, local_pair)
                end
            end
            sequences[sequence_index] = (merged_sequence, frequency)
        end

        get(pair_counts, pair, 0) == 0 || error("selected BPE pair was not depleted")
        delete!(pair_counts, pair)
        delete!(pair_sequences, pair)
        delete!(affected_pairs, pair)
        for affected_pair in affected_pairs
            count = get(pair_counts, affected_pair, 0)
            count > 0 && _heap_push!(
                candidate_heap,
                _BPECandidate(affected_pair, count),
                vocab_internal,
            )
        end
    end

    vocab = Dict{Int,Vector{UInt8}}(
        identifier => collect(bytes) for (identifier, bytes) in vocab_internal
    )
    merges = [(collect(left), collect(right)) for (left, right) in merges_internal]
    return vocab, merges
end
