# Writing style notes

These are defaults for editing the user's prose.
Explicit instructions for a particular piece take precedence.

## Voice and editing

- Preserve the author's voice and sentence structure when making small fixes.
- Prefer human, direct phrasing over polished-but-generic explanatory prose.
- Do not restate a motive the piece has already established.
- Describe other people's tools as answers to different workflows, not as failed
  attempts to converge.
- Do not explain a joke after making it.
- Do not assume a technical audience is completely new to the subject.
- Keep connected reasoning in natural-sized paragraphs. Avoid the ChatGPT habit
  of giving every turn of thought its own miniature paragraph.
- Treat author-provided titles and section headings as fixed unless asked to change them.
- Do not introduce em dashes merely to recast a sentence.
- Semicolons are part of the author's style. Preserve them; warn if a passage
  genuinely overuses them, but do not silently replace them.

## Source formatting

- Break prose lines at punctuation or meaningful syntactic boundaries.
- Balance semantic breaks with visual rhythm: adjacent lines should usually be
  roughly comparable in length.
- Around 80 characters is comfortable; 100 is a practical ceiling for prose.
- Do not split cohesive phrases into choppy fragments merely to meet a width.
- Long indivisible Markdown links and reference URLs may exceed the prose ceiling.
- In a formatting-only pass, do not change content. Compare against the staged
  version after normalizing soft line breaks, and verify fenced blocks exactly.

## References

- Never discard a reference link.
- If a link no longer belongs naturally in the prose, retain it in a reference
  section at the bottom.
- Prefer attaching links to names already present over adding prose merely to
  introduce sources. When several same-named tools share a collective phrase,
  fragments of that phrase may carry separate links.

## `twit-idea.md`

### Voice

- This is a personal technical essay, not a tutorial, manifesto or product
  comparison. Keep the opening autobiographical and self-mocking; let the
  technical passages become direct once the personal motive is established.
- Use first person for actual history, taste and naming decisions. State Git
  behaviour, commands and maintenance advice without “I want”, “I will” or
  “I would”. Do not make the author sound aggrieved about ordinary ergonomics.
- Keep the jokes dry and incidental. Odd names, Boyi and Shuqi, buds, rented
  slots and automatic janitors should not receive explanatory follow-up.
- Preserve concise colloquial constructions when they sound natural in the
  essay. “Ignored files are where ...” is fine; generic textbook expansion is
  not automatically an improvement.
- Avoid abstract concluding language such as “settle the convention without
  settling on a manager”. Prefer the concrete observation that produces the
  conclusion.

### Flow and proportion

- Each section should make one move. The Git introduction establishes the
  traditional sibling and bare layouts; the harness section surveys current
  uses; “not content” assembles the umbrella compromise; the jj section explains
  why *workspace* cannot name the umbrella; the final section chooses names and
  shows the resulting practice.
- The jj section deliberately gives Git-familiar readers a short account of
  changes, stacks and colocation before arriving at *workspace*. Tighten it
  locally; do not reduce it to a terminology footnote.
- The long naming table is the controlled version of the naming digression.
  Its blank comments and increasingly strange candidates are intentional; do
  not collapse it to a sensible shortlist.
- Do not insert research at the first technically related sentence. Place it
  only where it advances the section's current move. In particular, the final
  section must not become a second survey of manager tools.
- Keep details proportional across tools. This post does not shill Worktrunk or
  dismiss other CLI/TUI tools; link them compactly and describe the different
  choices they make.
- Preserve cuts. Do not restore compatibility-first warnings, dirty-repository
  migration advice, dot-directory caveats or generic package/build-cache
  guidance merely because they are true.
- Prefer a short, useful paragraph or no paragraph. Do not replace deleted
  material with another summary of the same motive.

### Technical exposition

- Assume familiarity with Git and worktrees. Teach a term when it matters to the
  layout, using the supported command that exposes it; do not turn the aside
  into an implementation-internals tour.
- Explain the common Git directory operationally: `.twit/.git/` holds shared
  state; `git rev-parse --git-common-dir` locates it, `--git-dir` locates the
  current worktree's administration, and `--git-path` resolves a specific path.
- Keep command examples usable in both Bash and PowerShell when possible.
  Commands and their explanations stay together; do not split a working sequence
  to manufacture another transition.
- Keep the ordinary-clone bootstrap and first `git worktree add` in one block.
  The lowercase `where origin/main ...` sentence follows it. `auth/` is the name
  seen in prompts and title bars; Git may call the branch `feat/oauth-retry`.
- The manager aside is concrete and small: a wrapper passes Git a directory,
  branch and optional starting point; Branchlet and Worktrunk merely show that
  existing tools can accommodate the layout.
- State maintenance and removal advice procedurally, without first person.
