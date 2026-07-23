<!-- Possible working titles -->

# A twit idea

<!--

# Think like a git, work like a twit

# Twit or twig

# How do you keep worktrees together?

-->

Vibe coding has made worktrees a must!
But I've found them useful even when I didn't have a coding plan subscription,
or had burned through my wallet by selecting an Opus on Fast through OpenRouter.
I'm somewhat of a yak-shaving die-hard; it's always tempting for me
to turn a transient idea into a wild goose chase;
at least being almost expelled from college didn't help me stop doing it.
So: worktrees. A gateway substance to let me feel
"oh surely it won't hurt; my working tree is there, ready to resume at any time..."

Boy am I gonna [die on that hill](https://en.wikipedia.org/wiki/Boyi_and_Shuqi "伯夷叔齐").

Codex was perfectly happy to put the worktree in its cache directory. I was not.
After getting annoyed by Codex desktop recently, I've decided to pin down how
I'm gonna use worktrees in the future, once and for all.

## What is a worktree?

A branch is not a folder. It is a movable name for a commit, and decades of
branch-switching commands making all the files in front of you change have done
Git users no favours here.

A *working tree* is the folder full of checked-out files. A normal clone already
has one. Git now calls that one the [**main worktree**](https://git-scm.com/docs/git-worktree);
`git worktree add` creates **linked worktrees**:

```console
$ git worktree add -b hotfix ../hotfix main
Preparing worktree (new branch 'hotfix')
```

`../hotfix` gets its own files, staging area, `HEAD`, and unfinished
merge/rebase state. It does not get another copy of the object database.
Its `.git` is a small text file pointing back into
`the-original-clone/.git/worktrees/...`.

That split is the bargain. Commits, local branches, tags, remote-tracking
branches, stashes, hooks, and most configuration are shared. Checked-out files
and the index are not. Commit something in one worktree and the object is
immediately visible in every other one; try to check out the same local branch
in two of them and Git normally refuses. The directory name and branch name have
no obligation to match—`auth/` may quite happily hold `feat/oauth-retry`.

So a worktree is cheaper than another clone, but it isn't a sandbox. Two
processes can still fight over a port, a database, a GPU, a branch ref, or the
meaning of "done". It only gives each of them a separate set of files and an
index. This happens to be exactly the kind of territorial dispute coding agents
get into most often.

Removal has one particularly nasty edge: `git worktree remove` bases its safety
check on `git status`. Ignored things can include the only copy of a `.env`, a
virtual environment, a build tree, or a model checkpoint; user configuration
can even make `status` omit ordinary untracked files. Before removing a
non-disposable tree, I want the boring ritual to enumerate all three classes:

```console
$ git -C ../hotfix status --short --branch --untracked-files=no
$ git -C ../hotfix ls-files --others --exclude-standard
$ git -C ../hotfix ls-files --others --ignored --exclude-standard
$ git worktree remove ../hotfix
$ git branch -d hotfix
```

The last command is separate on purpose. Removing a worktree does not remove its
branch, and `branch -d` gets a chance to object if it has not been merged.

## Why do harness people hype about it?

For a human, a second worktree means the half-debugged thing can remain
half-debugged while an urgent fix happens elsewhere. The editor stays open, the
breakpoints stay put, and nobody has to remember whether the useful uncommitted
change is in stash number three.

For an agent harness, it means five workers can all believe they own the
repository without immediately rewriting one another's files. The old feature
has acquired a new customer. GitHub's recent
[worktree introduction](https://github.blog/ai-and-ml/github-copilot/what-are-git-worktrees-and-why-should-i-use-them/)
explicitly credits AI for the sudden popularity, demonstrates a sibling
`../hotfix-workspace`, and then concedes that dependencies and folders pile up.
Quite. Where are all these things supposed to live?

Everyone agrees on `git worktree add`; agreement ends before the path argument.
[Claude Code](https://code.claude.com/docs/en/worktrees) defaults to
`.claude/worktrees/` inside the checkout. Codex defaults to
`$CODEX_HOME/worktrees`; to be fair, its current
[worktree settings](https://learn.chatgpt.com/docs/environments/git-worktrees)
let the user change that root and create permanent worktrees as projects.
[`gwq`](https://github.com/d-kuro/gwq) grows a global
`~/worktrees/host/owner/repo/branch` forest.
[Treehouse](https://github.com/kunchenguid/treehouse) leases numbered,
detached-HEAD slots and keeps their dependency caches warm.

[Worktrunk](https://github.com/max-sixty/worktrunk), the largest specialist
wrapper I found, defaults to siblings named like `repo.feature-auth` and
computes the path from the branch. It also copies caches, runs hooks, assigns
ports, launches agents, and removes the debris afterwards. Its `wt` command
collides with Windows Terminal, so the Windows package exposes `git-wt`. That
accident at least makes `git wt` a proper external Git subcommand; the namespace
is cursed, not technically wrong.

There are many more: `git gtr`, Branchlet, at least two serious Groves plus
several unrelated namesakes, several TUIs, and a fresh wrapper every time
somebody gets tired of typing the branch name twice. None has won the directory
argument. They are not even managing the same kind of thing:

- a durable checkout belonging to a branch;
- a disposable directory belonging to a task or chat;
- a reusable execution slot whose contents will be reset;
- a peer controlled by a bare repository somewhere in the middle.

I care about the first one, with occasional visits from the second. A cache is
the correct home for a rented stall. It is a bad home for a line of thought I
expect to resume after I've forgotten why I started it.

## But I'm not content with that...

Git refuses to choose a location. Most tutorials put `repo-hotfix` beside
`repo`, which works until the parent directory looks like somebody dropped a
box of branches on the floor. Putting linked worktrees *inside* the main
worktree is neater to the eye but makes the repository contain its own copies;
now an ignore rule and every recursive indexer have to know about the trick.
A global pool solves the spill by hiding it somewhere else.

While I was bikeshedding this, Nicholas Zakas published a
[gentle introduction](https://humanwhocodes.com/blog/2026/07/introduction-git-worktrees/)
that calls direct siblings and a shared sibling container the two conventions.
He prefers `my-project.worktrees/feature-name`. Good: the extra roof is already
unsurprising. I merely want to pull the original clone under it too.

This wasn't invented by agents. Morgan Cugerone described a tidy
[bare-repository worktree layout](https://morgan.cugerone.com/blog/how-to-use-git-worktree-and-in-a-clean-way/)
in 2021, and an older
[layout note](https://gist.github.com/sellout/3361145fac9bf2dfdc6a9bc18dcdff36)
asks almost exactly the same question. The bare controller is beautiful: the
repository itself sits at the centre and every checkout is a peer.

I still don't want it as the default.

A bare clone is not merely a normal clone with the files swept away. In
particular, [`git clone --bare`](https://git-scm.com/docs/git-clone.html)
copies remote heads directly into local branch refs and omits the usual
`origin/*` remote-tracking setup. More prosaically, IDEs, Git GUIs, hook setup,
Git LFS, submodules, language tools, and arbitrary project scripts are happiest
when handed an ordinary checkout. A bare controller can still serve linked
worktrees, but it cannot itself be the checkout those tools open or build.
Every individual inconvenience is soluble; together they are an entrance exam
I don't need.

I want one conventional clone that a tool can encounter without being taught my
religion. It should contain the real `.git/`, remain a useful checkout, and act
as the administrative anchor for every linked worktree. I just don't want it to
occupy the project's obvious name, because the project's obvious name is more
useful as a roof over all of them.

That costs one extra directory:

```text
somerepo/
├── .prime/          # ordinary clone; main worktree; the real .git/ lives here
├── dev/             # a linked worktree
├── auth-retry/      # another linked worktree
└── parser-probe/    # perhaps temporary, but not lost in somebody else's cache
```

`somerepo/` is not a repository. This is deliberate. Open `.prime/` or one of
its siblings in the editor; don't register the umbrella itself as a coding-agent
project and then act surprised when the agent regards the whole household as
one writable blob. A leading dot is a sign saying "infrastructure", not an
access-control mechanism. On Windows a leading dot does not make the folder
hidden at all.

I intend to keep the umbrella nearly empty. Project scripts and instructions
belong in the repository, so every worktree receives them. A hand-written note
or multi-root editor file may live one level up, but then it is ordinary
untracked local data and needs an ordinary backup. Git will not save a file it
cannot see, however pleasing the directory diagram.

The ordinary commands remain ordinary:

```powershell
New-Item -ItemType Directory somerepo
git clone REMOTE somerepo/.prime
Set-Location somerepo/.prime

git fetch origin
git worktree add -b feat/auth ../auth origin/main
git worktree list
```

Substitute the repository's actual default branch for `origin/main`. If every
Git that will touch the repository is 2.48 or newer, I may also opt into
relocatable links before adding siblings:

```powershell
git config worktree.useRelativePaths true
```

An existing, unoccupied branch is just:

```powershell
git worktree add ../auth feat/auth
```

The worktree directory is named for what I want to see in a shell prompt or
window title. The branch can keep whatever taxonomy the repository imposes.
And because `.prime/` is the main worktree, `git worktree remove` refuses to
remove it; `Remove-Item` remains quite capable of ignoring the symbolism.

For a repository with no remote yet:

```powershell
New-Item -ItemType Directory -Force somerepo/.prime | Out-Null
git -C somerepo/.prime init -b main
# Add the first files and commit them before creating linked worktrees.
```

A lone ordinary clone can be moved under a new umbrella without changing what
it is. I would not advertise a clever one-liner for an existing, dirty
collection of linked checkouts: first account for local-only refs, uncommitted
changes, and ignored files. Either rebuild from a fresh clone after preserving
all of them, or move the household and treat `worktree repair` as part of the
move, not an optional afterthought.

## Break a leg with じゅじゅつ

[Jujutsu](https://docs.jj-vcs.dev/latest/) calls its additional checkouts
**workspaces**. Its glossary goes as far as saying that a workspace is
[what Git calls a worktree](https://docs.jj-vcs.dev/latest/glossary/#workspace).
That is an analogy, not file-format compatibility.

A linked Git worktree contains a `.git` pointer. A secondary jj workspace
contains a `.jj/` directory whose `repo` file points back to the initial jj
repository. Current jj flatly refuses to initialise a colocated repository
inside a linked Git worktree: "Cannot create a colocated jj repo inside a Git
worktree."
Conversely, under the neutral umbrella described here, `git status` in a
jj-only secondary workspace finds no repository. They can be neighbours;
neither tool adopts the other's working copies.

The one bridge is the main checkout. It may be a
[colocated Git/jj repository](https://docs.jj-vcs.dev/latest/git-compatibility/),
with both `.git/` and `.jj/`. Both systems then use the same Git object store,
and jj automatically imports and exports supported Git refs when commands run
in that colocated checkout. From the common anchor I can create either kind of
sibling:

```text
somerepo/
├── .prime/          # .git/ and, optionally, .jj/
├── git-linked/      # .git pointer; Git manages this working copy
└── jj-second/       # .jj/ with a repo pointer; jj manages this working copy
```

This is not just dotfile pedantry. A Git worktree owns an index and usually a
branch or detached `HEAD`. A jj workspace owns a working-copy commit, shown as
`workspace-name@`; bookmarks are repository-wide names, not "the current
branch". If another jj workspace changes the tree of that commit, the files can
become stale; jj then asks for `jj workspace update-stale`. Git's
one-branch-per-worktree guard is solving a different problem.

The asymmetry continues when it is time to throw one away:

```console
$ jj workspace add ../jj-second --name jj-second
$ jj workspace list
$ jj workspace forget jj-second
```

`forget` stops tracking the workspace but intentionally leaves the directory
alone. Git's `worktree remove` removes the directory but intentionally leaves
the branch alone. I appreciate that both tools make deletion a two-part thought,
even though they disagree about which half to perform.

Tools such as [Treq](https://treq.dev/docs/concepts/workspaces/) build a managed
Git/jj hybrid above this. They demonstrate that the combination is useful; they
do not make `git worktree` and `jj workspace` interchangeable. This is the one
case where a neutral umbrella earns its rent without argument.

## See the grove and have the trees too

Here is the whole alleged innovation:

```text
<projects>/
└── somerepo/                 # umbrella; normally just the repository basename
    ├── .prime/               # conventional clone and main Git worktree
    ├── dev/                  # long-lived linked worktree, if wanted
    ├── feat-auth/            # task-named Git worktree
    ├── try-new-parser/       # another Git worktree
    └── jj-type-inference/    # possibly a jj workspace; inspect its dotfile
```

If the naked basename has to coexist with an already conventional checkout,
`somerepo.grove/` is a tolerable decorative variation. I don't think a mandatory
`worktrees/` or `workspaces/` layer buys anything unless both Git and jj copies
need to be visually sorted. The children already know what they are. So do
`git worktree list` and `jj workspace list`.

When two remotes have the same basename, `owner-somerepo/` or
`host.owner.somerepo/` can disambiguate the umbrella. The naked basename is a
default, not another law to litigate.

I am not turning this into six package-manager articles. The rule is to share
download caches and isolate mutable builds. Here, for example, each Python
worktree gets its own `.venv` while uv keeps sharing its package cache. The same
distinction applies to a Julia depot versus a project, Cargo's registry versus
`target/`, or a package-manager store versus `node_modules`. I will not symlink
two concurrently mutating build trees together merely because the immutable
downloads were good neighbours.

The default hooks directory and ordinary Git configuration are already shared
through the common `.git`; `core.hooksPath` can of course point somewhere else.
When a setting really belongs to one worktree, enable
`extensions.worktreeConfig` and use `git config --worktree`; don't infer the
current checkout by scraping the `.git` file. Git provides
`git rev-parse --git-dir`, `--git-common-dir`, and `--git-path` precisely
because `.git` is not always a directory. Enabling that extension is a
repository-format choice: older Git versions refuse to open it, so it is not
part of my bootstrap.

For removal, inspect tracked, untracked, **and ignored** material, then use the
porcelain command. If a directory was definitely deleted behind Git's back,
override Git's normal three-month grace period explicitly:

```powershell
git -C .prime worktree prune --dry-run --verbose --expire now
git -C .prime worktree prune --verbose --expire now
```

If the whole umbrella was moved behind Git's back, give `repair` the new path
of every linked worktree:

```powershell
git -C .prime worktree repair ../dev ../feat-auth ../try-new-parser
```

`prune` removes stale administrative records, not live worktree directories.
`repair` is for moved umbrellas and pointers. Hand-editing
`.prime/.git/worktrees/` is a fine way to turn a directory convention into an
archaeology project. Nor should an eager cleanup script run
[`git gc --prune=now`](https://git-scm.com/docs/git-gc.html) while several
agents may be writing objects; Git's normal grace period exists for a reason.

Relative worktree links make moving the whole grove less dramatic, but the
setting shown above enables another repository extension which old Git refuses
to open. Absolute paths plus `repair` are the compatibility-first choice;
relative paths are attractive once every machine involved is new enough.

Submodules remain the conspicuous exception. Git's own
[worktree manual](https://git-scm.com/docs/git-worktree#_bugs) still calls their
multiple-checkout support incomplete. A linked worktree containing submodules
cannot be moved with `git worktree move`, and even a clean one needs `--force`
to be removed. I would not let an automatic janitor learn about that flag.

What should the main clone be called? `root` is accurate until an agent sees the
word and gets ideas. `anchor` and `control` sound like infrastructure products.
`repo` says nothing. Repeating `somerepo/somerepo` is prosaic, which is not a
vice, and gives editor windows a wonderfully normal title.

The dot-directory tournament was less dignified. `.init` describes an event,
and `.seed` sounds like an RNG. `.master` carries both historical luggage and a
false suggestion about the checked-out branch. `.self` is appealingly weird;
`.demiurge` is more honest if the clone is going to create and destroy worlds.
The authoritarian bench—`.dictator`, `.stalin`, `.vader`, and friends—was
memorable but would make error messages needlessly alarming.

`.twit` is the better joke and appears not to collide with a worktree tool
(only an old Node Twitter client). `.prime` is the name I would still understand
before coffee. It suggests the primary checkout and the prime mover without
claiming to be the repository itself, and it beats `.init` cleanly. So `.prime`
wins; `.twit` gets the title and reserve duty.

This is not a Git standard. It is one `mkdir`, one oddly named normal clone, and
an agreement about where the furniture goes. If the post goes viral, I will
pretend the name emerged by consensus. If it doesn't, at least future me no
longer gets to reopen the naming tournament.

