# Work log

## 2026-07-23

- 23:03:09 +08:00 — Started expanding `twit-idea.md` from the user's opening and outline on branch `writing/twit-idea`. Confirmed the worktree is clean at commit `f18f3ee`.
- 23:03:09 +08:00 — Set the writing constraint: preserve the user's cynical, self-implicating voice; avoid generic explanatory-blog cadence; use documentation and executable probes for factual claims.
- 23:06:45 +08:00 — Split research into three parallel tracks: native Git and current harnesses; Jujutsu workspaces/colocation; wrapper ecosystem and voice-aware structure.
- 23:06:45 +08:00 — Checked current Codex worktree documentation. The app defaults managed worktrees to `$CODEX_HOME/worktrees`, but now exposes a configurable worktree root and distinguishes disposable managed worktrees from permanent worktrees.
- 23:06:45 +08:00 — Probed the proposed umbrella layout in a temporary repository: `somerepo/.prime` can hold the conventional clone and real `.git`, a sibling `somerepo/topic-one` links back to it, and `git` correctly reports that `somerepo/` itself is not a repository.
- 23:06:45 +08:00 — Found a timely independent comparison: Nicholas Zakas documents direct-sibling and shared-sibling-container conventions; his preferred `project.worktrees/` structure makes the umbrella layer unsurprising, while leaving our conventional-clone anchor and mixed Git/jj case distinct.
- 23:13:39 +08:00 — Completed the first full article draft (roughly 2,270 words), filling every outline section: Git's model, agent/wrapper motivations, the conventional-clone argument, exact Git/jj non-interoperability, the proposed grove, dependency isolation, lifecycle, and the naming tournament.
- 23:13:39 +08:00 — Incorporated primary-source corrections: GitHub Desktop now supports worktrees; Worktrunk's Windows `git-wt` executable can function as an external Git subcommand; current Codex lets users relocate its managed-worktree root.
- 23:13:39 +08:00 — Verified all article links resolve except Treq's page through direct open; a search-engine fetch of the same canonical URL returned the current page. `git diff --check` and the staged diff check are clean.
- 23:13:39 +08:00 — Requested adversarial follow-up reviews of Git accuracy, Jujutsu accuracy, and LLM-like prose patterns. No repository-writing work was delegated.
- 23:19:06 +08:00 — Applied the adversarial review findings: corrected `.jj/` pointer wording and stale-workspace semantics; separated `prune` from `repair`; made Codex, bare-controller, and hook claims narrower; added remote/default-branch, migration, basename-collision, submodule, relative-link, and old-Git compatibility qualifications.
- 23:19:06 +08:00 — Strengthened removal safety. The draft now explicitly enumerates tracked changes, ordinary untracked files, and ignored files; uses `--expire now` only after confirming a worktree was actually deleted; and warns against automatic forced cleanup around submodules.
- 23:19:06 +08:00 — Reduced generated-sounding connective tissue and collapsed the ecosystem dependency matrix into one rule with a few relevant examples. Structural checks report 12 balanced code-fence pairs, balanced HTML comments, no outline placeholders, and a clean `git diff --check`.
- 23:19:06 +08:00 — Two disposable probes remain under the user temp directory because the approval reviewer errored on cleanup: `twit-worktree-probe-fc4c82695d974532b17c4e1614ebda5c` and `jj-workspace-probe-a904cebab9bb4cad91c11dd9dee7e4b0`. They contain only synthetic local repositories and are not project material.
- 23:20:10 +08:00 — Completed the final mechanical audit on branch `writing/twit-idea`. Only `twit-idea.md` and `work-log.md` differ from branch tip; all five outline headings have substantive bodies; 12 code-fence pairs and both HTML comments are balanced; 18 source links are present; all six PowerShell blocks parse successfully; no placeholder remains; `git diff --check` is clean. Final draft length is roughly 2,786 words.
- 23:20:10 +08:00 — Commit status: the first version of `work-log.md` was staged successfully, but the approval reviewer errored when authorizing `git commit` and again when asked to refresh the staged article/log. The completed article and later log entries therefore remain unstaged; no workaround was attempted without fresh explicit approval.

## 2026-07-24

- 00:13:13 +08:00 — Reworked the full article after author review. Removed the beginner Git tutorial, kept the official manual's intended worktree uses at an experienced-reader altitude, and moved process/resource-isolation caveats into the harness discussion.
- 00:13:13 +08:00 — Reconstructed the actual argument from the preceding design conversation: durable human-owned thought versus disposable harness state; the project-basename namespace consumed by the first conventional clone; direct-sibling spillage; the appeal and compatibility tax of a bare controller; and the reason for nesting a normal main checkout beneath a neutral umbrella.
- 00:13:13 +08:00 — Reserved the concrete `.prime` layout, bootstrap commands, local naming, editor/agent boundary, dependency isolation, maintenance, removal safety, repair, submodule caveat, and naming tournament for the final section. Kept all five author-provided section titles verbatim after briefly changing and then restoring the first.
- 00:13:13 +08:00 — Preserved all 18 distinct source URLs in prose and duplicated the complete set in a bottom reference section so later cuts cannot silently discard research.
- 00:16:27 +08:00 — Completed a sentence-level voice pass after the author approved the new framing and tone. Kept that framing fixed; made the Jujutsu and closing sections continue the same dry, personal register instead of reading like a detached operations appendix.
- 00:16:27 +08:00 — Made the ownership complaint more explicit: agent-managed disposable execution is not the same thing as a human-owned unfinished line of thought; the first conventional clone consumes the project basename; a bare controller is cleaner but exacts a compatibility tax.
- 00:16:27 +08:00 — Final current-state audit: all five author headings remain exact, 18/18 original URLs remain with none added or lost, all seven PowerShell blocks parse, nine code blocks are balanced, and `git diff --check` passes.
- 00:19:32 +08:00 — Incorporated line-level author feedback on the second voice pass: restored the calmer manual paragraph and section transition from the prior version, while retaining the approved concise description of shared versus per-worktree state. Rechecked the diff for whitespace errors.
- 00:31:45 +08:00 — Clarified the opening chronology: worktrees were already useful before the current AI-coding economy, rather than merely during periods when the author lacked a subscription or had exhausted one. Preserved the grammatical corrections to the model and service names.
- 00:36:54 +08:00 — Simplified that opening again after author review: removed the added em-dash construction and returned to the intended original sentence structure, changing only tense, parallelism, the article before `Opus`, and `OpenRouter` capitalization.
- 00:38:47 +08:00 — Restored the opening's general agent-harness versus human-developer contrast. Qualified it with “often” and “tend to” because surveyed tools also use in-repository managed directories, global pools, and ordinary siblings; retained Codex as the specific incident rather than the whole subject.
- 00:40:04 +08:00 — Removed the survey-like qualifiers from that contrast after further author review. Recast it as two plain observations—harnesses are happy to bury worktrees in caches; people want them near the primary clone—while leaving the later section to supply the factual qualifications.
- 00:42:18 +08:00 — Simplified the agent sentence to the author's original construction and retained the approved human sentence unchanged.

### Pre-compaction context promotion

- 13:28:44 +08:00 — The design discussion began well before this log was
  requested. The following entries deliberately reconstruct the decisions that
  must survive compaction; they are more authoritative than early generated
  prose, but not an exhaustive transcript.
- The post is the author's personal decision about a filesystem convention, not
  a proposed Git standard and not a wrapper comparison. Do not add an
  obvious “this is not a standard” disclaimer, explain the title joke, or make
  the author defend the existence of the preference again.
- Keep all five author-supplied section titles verbatim:
  `What is a worktree?`, `Why do harness people hype about it?`,
  `But I'm not content with that...`, `Break a leg with じゅじゅつ`, and
  `See the grove and have the trees too`.
- The fundamental distinction is ownership and lifetime. A cache, chat ID, or
  reusable slot is an appropriate home for disposable harness execution. A
  durable unfinished line of thought is human-owned project state and should
  not silently inherit a product's cache location, eviction policy, naming, or
  lifetime. This distinction has already been established in the opening and
  second section; the third section must not explain the motive again.
- The first section is for Git's own mental model and established layouts at an
  experienced-reader altitude. It now covers ordinary direct siblings and the
  bare-controller purist layout. Do not move that survey into the load-bearing
  third section.
- Morgan Cugerone's post deserves more than a passing “prior art” link when the
  first section is revisited. Its distinctive layout is an umbrella containing
  `.bare/`, a top-level `.git` file pointing to `./.bare`, and peer worktrees.
  That makes the umbrella an administrative Git entry point. The post's later
  update reports friction around fetching remote branches and provisioning
  dependencies. Use it as a concrete alternative and foil, not as proof that
  bare layouts are obsolete.
- Bare-controller compatibility is no longer a severe universal problem:
  respectable current tools generally understand bare repositories and linked
  worktrees. The conventional clone remains attractive as a compatibility
  baseline and a useful checkout, not because modern tools categorically fail
  without one.
- `git clone --mirror` is not the purer answer for a development household.
  It implies `--bare`, maps all source refs, and configures updates to overwrite
  them. That replication contract is inappropriate when local branches are
  active worktree state. Rewriting the refspec would effectively stop treating
  the clone as a mirror.
- The current second section is substantially approved. It treats agent
  worktrees as cheap file/index separation while acknowledging shared refs and
  non-Git resources; distinguishes a possession from a room key; and presents
  Claude Code, Codex, `gwq`, Treehouse, Worktrunk, `git gtr`, Branchlet,
  multiple Groves, LazyWorktree, Wisetree, rust-git-worktree, and forestui as
  respectable tools serving different workflows.
- Do not divide those tools into “CLIs” versus “TUIs”; interface style is not
  the relevant category. The meaningful axes are path placement, naming,
  copied/setup state, and how much lifecycle the tool owns. Describe other
  people's work as valid answers to different workflows, not failed attempts
  to converge or products the article is ranking.
- The author's current transition out of that survey is:
  “In all practical terms, worktree (and task agent) orchestration / is a solved
  problem, albeit with a fragmented landscape.” Do not eagerly rewrite it.
- Links should carry attribution without bloating the prose. Link every named
  tool that can be identified. Several unrelated tools sharing the name Grove
  may be linked by splitting “several Groves” across syllables; do not use that
  device to turn an interface category such as “TUIs” into a pseudo-name.
  Preserve every accumulated reference, moving displaced links to the bottom
  reference section.
- The next task is the load-bearing third section. Its current text is
  scaffolding, not approved prose. In particular, the detailed Codex child-task
  anecdote, the six-weeks-later inventory of motives, and the repeated
  bare-repository explanation should not survive merely because they are
  already written.
- The third section's actual starting insight is older and simpler than agents:
  `git clone ... somerepo` spends the project's best local name on one working
  copy. Once durable worktrees appear, direct siblings require repeated
  repository prefixes, nesting inside the privileged checkout gives recursive
  tools and ignore rules a false household root, and a global/tool-owned pool
  moves the household out of sight.
- Nicholas Zakas supplies the shared-container roof
  (`my-project.worktrees/feature-name`). The useful next question is why the
  original checkout remains outside that roof. The bare layout supplies peer
  symmetry. The author's compromise combines those established ideas: the
  naked repository basename names the whole local household; every checkout is
  a child; one child remains a conventional clone with the real `.git/` and a
  useful main worktree.
- This collection is not presented as an original invention. Attribute the
  direct-sibling/shared-container convention, bare-controller layouts, and the
  separation of directory names from branch names. The author's contribution
  is collecting the ideas into one pattern because they need to settle on a
  personal convention from now on.
- Once the basename belongs to the household, child directory names may describe
  human lines of thought rather than repeat repository branch taxonomy:
  `auth/` may contain `feat/oauth-retry`. Preserve this idea with attribution;
  do not claim Git's branch/path independence as novel.
- Naming history matters. `somerepo.grove/` is intuitive for the umbrella, but
  “grove” is already used by many worktree tools precisely because it is
  intuitive. It is therefore demoted to a spare stylistic variation, not
  rejected. The default umbrella is the naked basename `somerepo/`. The
  conventional clone eventually becomes `.prime/`; `.twit` remains the title
  joke and reserve option.
- The third section should stop after establishing the abstract compromise.
  The later Jujutsu section explains why Git worktrees and jj workspaces may be
  neighbours but are not interchangeable. The final section owns the concrete
  `.prime` tree, bootstrap commands, naming exceptions, editor/agent boundary,
  dependency isolation, per-worktree configuration, removal safety, repair and
  pruning, submodule caveat, and naming tournament.
- Source style remains semantic rather than rigidly column-wrapped: break at
  punctuation or meaningful clauses, keep adjacent lines visually comparable,
  aim around 80 characters and treat 100 as a practical prose ceiling, while
  allowing indivisible links to exceed it. Preserve the author's semicolons,
  sentence structure, dry personal tone, technical altitude, and jokes.
