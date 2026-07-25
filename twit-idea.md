<!-- Possible working titles -->

# A twit idea

<!--

# Think like a git, work like a twit

# Twit or twig

# How do you keep worktrees together?

-->

Vibe coding has made worktrees a must!
But I'd found them useful even before I had a coding plan subscription
or accidentally burnt through my wallet by selecting Opus on Fast through OpenRouter.
I'm somewhat of a yak-shaving die-hard;
it's always tempting for me to turn a transient idea into a wild goose chase;
at least being almost expelled from college didn't help me stop doing it.
So: worktrees. A gateway substance to let me feel
"oh surely it won't hurt; my working tree is there, ready to resume at any time..."

Boy am I gonna [die on that hill](https://en.wikipedia.org/wiki/Boyi_and_Shuqi "伯夷叔齐").

Agent harnesses prefer to keep worktrees in a cache directory;
human devs usually want them somewhere near the primary clone.
After getting annoyed by Codex desktop recently,
I've decided to pin down how I'm gonna use worktrees in the future, once and for all.

## What is a worktree?

The [manual](https://git-scm.com/docs/git-worktree) gives worktrees a modest job:
keep more than one branch checked out at once.
A clone or `init` supplies the main worktree;
`git worktree add` attaches more of them to the same repository.
Git also imagines a temporary detached worktree for an experiment,
and provides `lock` for a worktree that may disappear with a removable disk or network mount.

Each worktree gets a `HEAD`, an index and its own half-finished rebase;
objects and ordinary refs are shared.
The directory and branch names need not match,
and Git doesn't give a rat about where the directory should go.

Traditionally, the main worktree keeps the repository's obvious name
and linked worktrees become its neighbours:
`somerepo/`, `somerepo-hotfix/`, `somerepo-new-parser/`.
Git's own `../hotfix` example points in exactly that direction.
It is simple, and the original clone remains an ordinary checkout;
it also lets one repository gradually colonize the parent directory.
Spoiler: make an umbrella directory; this post is really just about
the trivial part: what to name it.

A purist can remove even the privilege of the main checkout:
make a [bare clone](https://git-scm.com/docs/git-clone#Documentation/git-clone.txt---bare) and make every working copy a linked peer.
Morgan Cugerone described a clean [bare worktree layout](https://morgan.cugerone.com/blog/how-to-use-git-worktree-and-in-a-clean-way/) in 2021,
and this older [layout note](https://gist.github.com/sellout/3361145fac9bf2dfdc6a9bc18dcdff36) was already worrying the same bone.
The symmetry is attractive.
The controller is no longer a checkout that an editor can open or a build can use;
tools written with an ordinary clone in mind must be pointed at a linked worktree.

That is about all the Git lesson this post needs.
The path argument was left to the caller.
My problem is with what callers have lately decided it means.

## Why do harness people hype it up?

For a human dev with a healthy life,
a second worktree lets the half-debugged thing remain half-debugged
while an urgent fix happens elsewhere.
The editor stays open, the breakpoints stay put,
and nobody has to remember whether the useful uncommitted change is in stash number three.
Agents did not invent [this use](https://github.blog/ai-and-ml/github-copilot/what-are-git-worktrees-and-why-should-i-use-them/);
but with token consumption being the best proxy indicator of welfare
that mankind can come up with, it's not at all surprising this is catching on.
One worktree per task gives each agent its own files and index without paying for another clone.
Refs remain shared; ports, databases, GPUs
and every other non-Git resource remain somebody else's problem.

The harness then has to decide whether its checkout is a possession or a room key.
[Claude Code](https://code.claude.com/docs/en/worktrees) keeps managed worktrees under `.claude/worktrees/`.
Codex has used `$CODEX_HOME/worktrees`;
its current [settings](https://learn.chatgpt.com/docs/environments/git-worktrees) also distinguish managed worktrees from permanent ones.
[`gwq`](https://github.com/d-kuro/gwq) builds a global `~/worktrees/host/owner/repo/branch` forest.
[Treehouse](https://github.com/kunchenguid/treehouse) leases warm numbered slots.
Many other worktree tools are emerging too:
[Worktrunk](https://github.com/max-sixty/worktrunk),
[`git gtr`](https://github.com/coderabbitai/git-worktree-runner),
[Branchlet](https://github.com/raghavpillai/branchlet),
[sev](https://github.com/thisguymartin/grove)[er](https://github.com/nicksenap/grove)[al](https://github.com/lost-in-the/grove) [Groves](https://www.usegrove.dev/),
[LazyWorktree](https://github.com/chmouel/lazyworktree),
[Wisetree](https://github.com/victorcorcos/wisetree),
[rust-git-worktree](https://github.com/ozankasikci/rust-git-worktree)
and [forestui](https://pypi.org/project/forestui/).
They make different choices about where a tree goes, what names it,
what gets copied into it and how much of its lifecycle the tool owns,
because they support different things:
a durable checkout, a throwaway chat, a reusable slot or a peer around a bare controller.

In all practical terms, worktree (and task agent) orchestration
is a solved problem, albeit with a fragmented landscape.

## But I'm not content yet...

The part I am not content with is older and dumber:
`git clone REMOTE somerepo` spends the best local name on one working copy.
The second durable checkout becomes `somerepo-auth`, the third `somerepo-parser`,
and soon the parent directory looks as if a repository burst
(again, not [my](https://humanwhocodes.com/blog/2026/07/introduction-git-worktrees/)
[original](https://morgan.cugerone.com/blog/how-to-use-git-worktree-and-in-a-clean-way/)
[idea](https://gist.github.com/sellout/3361145fac9bf2dfdc6a9bc18dcdff36)).
A global pool only moves the decision out of sight.

Nicholas Zakas calls direct siblings and a shared sibling container
the two conventions in his recent [introduction](https://humanwhocodes.com/blog/2026/07/introduction-git-worktrees/),
and prefers `my-project.worktrees/feature-name`.
I like the roof; leaving the original clone beside it
still gives one local project two external entry points.
I want the original checkout beneath the roof as well,
so the obvious basename means the whole household.

The bare layout from the previous section gets that symmetry by doing without a main worktree.
Modern tools generally cope with it,
but an ordinary clone is still a useful checkout
and the compatibility option that asks least of the next tool.
I want the symmetry without giving that up.

So, keep the ordinary clone, but demote it mentally.
It remains the main worktree, with the real `.git/`
and all the habits of a conventional clone.
Linked worktrees become its siblings under a neutral umbrella.
The original checkout remains administratively special;
it just no longer gets to name the whole local project.

I see no need for a compulsory `worktrees/` layer beneath the umbrella.
Direct children are the default; another layer can earn its place later.

As many have noted,
Git does not restrict the worktree [path to its branch name](https://git-scm.com/docs/git-worktree).
A durable tree still deserves a useful name,
so `auth/` may contain `feat/oauth-retry`;
an ad-hoc one can keep whatever was convenient.

We now have a rough idea that combines
the shared-container convention Zakas describes,
the peer checkouts of bare layouts, and so on.
What remains is just to decide how to assemble them
into the convention I mean to use from now on.

## Break a leg with じゅじゅつ

[Jujutsu](https://docs.jj-vcs.dev/latest/) is an emerging VCS.
The first thing you'll notice is that there is no staging area,
and the second is that it does not care about 'branches' like Git
(what Git calls branches are physiologically [buds](https://en.wikipedia.org/wiki/Bud) on a plant).
But most of its appeal lies in how casually it lets you edit a stack of changes.
The working copy belongs to the change being edited;
`jj` snapshots it into a new commit at the start of almost every command.

What makes a branchless and stackful life easy is the new layer of abstraction
called [changes](https://docs.jj-vcs.dev/latest/glossary/#change).
Each change keeps a stable identity as its underlying commit is rewritten.
Edit one in the middle of a stack and jj rebases its descendants automatically;
[conflicts](https://docs.jj-vcs.dev/latest/conflicts/) can remain in commits
rather than being an interruptible operation,
and the [operation log](https://docs.jj-vcs.dev/latest/operation-log/)
makes the rearrangement inspectable and undoable.

Jujutsu is very Git-compatible at the moment:
a Git-backed workspace created by current jj is
[colocated by default](https://docs.jj-vcs.dev/latest/git-compatibility/#colocated-jujutsugit-workspaces):
`.git/` and `.jj/` sit beside each other,
both tools use the same working copy,
and jj imports and exports Git state as it runs.

I won't make this any more of a jj shill than I already have.
See [Justin's cheatsheet](https://justinpombrio.net/2025/02/11/jj-cheat-sheet.html#:~:text=cheat%20sheet%20for%20Jujutsu);
alternatively, the [official docs](https://docs.jj-vcs.dev/latest/git-comparison/)
compare against Git concept by concept.
[Evan Martin's tutorial](https://evmar.github.io/jjtut/) is concise and aimed at Git users,
while [Steve Klabnik's](https://steveklabnik.github.io/jujutsu-tutorial/) is more conversational.

What matters for us, though, is jj's counterpart to a worktree.
A jj *[workspace](https://docs.jj-vcs.dev/latest/glossary/#workspace)*
is one working copy and its associated repository.
They can live alongside worktrees,
though neither tool has much of a clue about
the other's additional checkout directories.
The umbrella may contain any number of Git worktrees or jj workspaces,
so `somerepo.workspace/` would be confusing to a jj user.
Colocating jj with Git in the primary clone
lets one checkout serve both tools without ceremony,
a convenience the bare-clone layout cannot offer.

That cooperation stops at the additional working copies.
Jujutsu still does not treat a linked Git worktree as a jj workspace.

So: the umbrella can contain the colocated child, Git worktrees and jj workspaces.
If both kinds multiply, intermediate `git/` and `jj/` layers can earn their place.
Otherwise direct children remain enough.
[Treq](https://treq.dev/docs/concepts/workspaces/) already manages one such hybrid;
my convention happens to leave room for it.

## See the grove and have the trees too

So we are just deciding the naming of the umbrella and the primary clone now.

People already use `somerepo.worktrees/` in the sibling-container way;
someone deleting old worktrees could delete the primary clone by accident
if one names the umbrella that way.
`somerepo.workspace/` is decent and weakly aligns with VS Code usage,
but I just like jj enough that I want to avoid its word for the counterpart of worktrees.
`somerepo.grove/` sounds intuitive enough to me.
Several worktree tools use the word *grove* too, but I don't think a collision is likely.
It remains a useful variation, though the basename already suffices for me.

I had a little too much fun naming the primary clone though:

<!-- https://stackoverflow.com/questions/17536216/create-a-table-without-a-header-in-markdown -->

| &nbsp;                    | &nbsp;                                                                                                       |
| ------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `<umbrella>/.repo`        | Closest to the plain arrangement example in `git worktree` documentation.                                    |
| `<umbrella>/somerepo`     | Repetitive in the path if the umbrella also uses the basename, but plays nice with editor titlebars.         |
| `<umbrella>/.somerepo`    | A dot-dir can suggest that this is not the checkout for daily work, and may even stay detached.              |
| `.root`                   | The agent makes a mistake with a shell command, and boom.                                                    |
| `.anchor`                 | Apt if this checkout is only administrative; less so if active development also happens in it.               |
| `.control`                | Similar.                                                                                                     |
| `.host`                   | The same sort of shell-command hazard as `.root`.                                                            |
| `.master` / `.main`       | Except one doesn't have to checkout the trunk here.                                                          |
| `.GIT`                    | Immediately collides with `.git` on ordinary Windows filesystems.                                            |
| `.init` / `.clone`        | Describes an event; perhaps too enigmatic?                                                                   |
| `.genesis`                | More enigmatic.                                                                                              |
| `.seed`                   | Sounds like an RNG.                                                                                          |
| `.zero`                   |                                                                                                              |
| `.nucleus`                |                                                                                                              |
| `.substrate`              |                                                                                                              |
| `.canopy`                 | There is already a [project](https://canopy.itsol.tech/) with this name.                                     |
| `.hermit`                 | There is already a [project](https://cashapp.github.io/hermit/usage/get-started/) with this name.            |
| `.self`                   |                                                                                                              |
| `.ansich`                 |                                                                                                              |
| `.monos` / `.monad`       |                                                                                                              |
| `.noumenon` / `.demiurge` |                                                                                                              |
| `.sith` / `.vader`        |                                                                                                              |
| `.dictator` / `.stalin`   |                                                                                                              |
| `.prime`                  | Conveys that this is either the OG init or the first clone; looks pretty good?                               |
| `.primary`                | Variant of `.prime`.                                                                                         |
| `.twit`                   | or `.twig`; whatever.                                                                                        |

`.prime` feels like the all-rounder to me,
though `.anchor` may convey your intention more clearly
when you're doing a bare-like experience.
`.twit`/`.twig` would be a good pun on Git itself though.

Next time, I am cloning into this:

```text
<projects>/
└── somerepo/                 # umbrella; normally just the repository basename
    ├── .twit/                # ordinary clone; main worktree; real .git/
    ├── dev/                  # a long-lived Git worktree
    ├── invoice-rounding/     # another Git worktree
    ├── parser-probe/         # perhaps temporary, but mine
    └── jj-type-inference/    # perhaps a jj workspace
```

I may also name the umbrella `owner.somerepo(.grove)/` or even `forge.owner.somerepo(.grove)/`
if I need disambiguation.

The primary clone is still an ordinary clone.
With Git 2.48 or higher, its worktree links can be relative from the start:

```console
mkdir somerepo
git clone REMOTE somerepo/.twit
cd somerepo/.twit

git config worktree.useRelativePaths true
git worktree add -b feat/oauth-retry ../auth origin/main
# or if `feat/oauth-retry` already exists:
# git worktree add ../auth feat/oauth-retry
git worktree list
```

where `origin/main` just stands for the actual remote branch you want.
Relative links make the umbrella more self-contained.
`auth/` is the name you see in a shell prompt or editor title;
Git can keep calling the branch `feat/oauth-retry`.

For a new repository with no remote,
Git can create the nested directory itself:

```console
git init -b main somerepo/.twit
# Make the first commit before adding linked worktrees.
```

If an umbrella with absolute links moved without Git's notice,
give `repair` every new worktree path:

```console
git worktree repair ../dev ../invoice-rounding ../parser-probe
```

The umbrella is not a repository,
so open `.twit/` or one of its siblings in editors and coding agents.
Give an agent the umbrella as its "project root"
and it may reasonably treat every checkout as in scope;
or give a subagent a specific worktree,
and you have slightly better isolation.

A small wrapper is enough to create worktrees in this layout:
pass a directory, a branch and, when needed, a starting point to Git.
Some existing tools fit too:
[Branchlet](https://github.com/raghavpillai/branchlet) can produce the same paths
with a template,
while [Worktrunk](https://github.com/max-sixty/worktrunk) can use worktrees
created directly by Git.

The primary clone's `.git/` is what Git calls the *common Git directory*:
it holds the objects, refs, ordinary configuration and default hooks
shared by every worktree.
`core.hooksPath` may put hooks somewhere else.
`git rev-parse --git-common-dir` finds the shared directory
from any worktree;
`git rev-parse --git-dir` instead finds that worktree's own
administrative directory.
If a setting really belongs to one worktree,
enable `extensions.worktreeConfig`
and use `git config --worktree`.
Scripts should use those commands or `git rev-parse --git-path PATH`
instead of assuming `.git` is a directory.

Before removing a worktree,
inspect more than `git status` normally shows:

```console
git -C ../auth status --short --branch --untracked-files=no
git -C ../auth ls-files --others --exclude-standard
git -C ../auth ls-files --others --ignored --exclude-standard
git worktree remove ../auth
git branch -d feat/oauth-retry
```

`prune` removes stale registrations, not live worktree directories.
If a directory was definitely deleted behind Git's back,
override the usual three-month expiry:

```console
git worktree prune --dry-run --verbose --expire now
git worktree prune --verbose --expire now
```

Submodules remain the conspicuous exception to automatic cleanup.
Git's own [worktree manual](https://git-scm.com/docs/git-worktree#_bugs)
still calls their multiple-checkout support incomplete.
A linked worktree containing submodules cannot be moved with `git worktree move`,
and even a clean one needs `--force` to be removed.
So don't let agent orchestration automate either operation.

---

**References accumulated while overthinking this**

- [Boyi and Shuqi](https://en.wikipedia.org/wiki/Boyi_and_Shuqi)
- [Git: `git-worktree`](https://git-scm.com/docs/git-worktree)
- [GitHub: What are Git worktrees and why should I use them?](https://github.blog/ai-and-ml/github-copilot/what-are-git-worktrees-and-why-should-i-use-them/)
- [Claude Code worktrees](https://code.claude.com/docs/en/worktrees)
- [Codex worktree settings](https://learn.chatgpt.com/docs/environments/git-worktrees)
- [`gwq`](https://github.com/d-kuro/gwq)
- [Treehouse](https://github.com/kunchenguid/treehouse)
- [Worktrunk](https://github.com/max-sixty/worktrunk)
- [`git gtr`](https://github.com/coderabbitai/git-worktree-runner)
- [Branchlet](https://github.com/raghavpillai/branchlet)
- [Grove: Zellij worktree workspaces](https://github.com/thisguymartin/grove)
- [Grove: multi-repository worktree workspaces](https://github.com/nicksenap/grove)
- [Grove: worktree and tmux management](https://github.com/lost-in-the/grove)
- [Grove: Claude Code session and worktree management](https://www.usegrove.dev/)
- [LazyWorktree](https://github.com/chmouel/lazyworktree)
- [Wisetree](https://github.com/victorcorcos/wisetree)
- [rust-git-worktree](https://github.com/ozankasikci/rust-git-worktree)
- [forestui](https://pypi.org/project/forestui/)
- [Canopy](https://canopy.itsol.tech/)
- [Hermit](https://cashapp.github.io/hermit/usage/get-started/)
- [Nicholas Zakas: An introduction to Git worktrees](https://humanwhocodes.com/blog/2026/07/introduction-git-worktrees/)
- [Morgan Cugerone: a bare-repository worktree layout](https://morgan.cugerone.com/blog/how-to-use-git-worktree-and-in-a-clean-way/)
- [Sellout's worktree layout note](https://gist.github.com/sellout/3361145fac9bf2dfdc6a9bc18dcdff36)
- [Git: `git-clone`](https://git-scm.com/docs/git-clone.html)
- [Git: `git clone --bare`](https://git-scm.com/docs/git-clone#Documentation/git-clone.txt---bare)
- [Jujutsu documentation](https://docs.jj-vcs.dev/latest/)
- [Jujutsu: comparison with Git](https://docs.jj-vcs.dev/latest/git-comparison/)
- [Evan Martin's Jujutsu tutorial](https://evmar.github.io/jjtut/)
- [Steve Klabnik's Jujutsu tutorial](https://steveklabnik.github.io/jujutsu-tutorial/)
- [Jujutsu glossary: workspace](https://docs.jj-vcs.dev/latest/glossary/#workspace)
- [Jujutsu: colocated Git repositories](https://docs.jj-vcs.dev/latest/git-compatibility/)
- [Jujutsu: colocated Git/jj workspaces](https://docs.jj-vcs.dev/latest/git-compatibility/#colocated-jujutsugit-workspaces)
- [Treq workspaces](https://treq.dev/docs/concepts/workspaces/)
- [Git: `git-gc`](https://git-scm.com/docs/git-gc.html)
- [Git worktree submodule caveat](https://git-scm.com/docs/git-worktree#_bugs)
