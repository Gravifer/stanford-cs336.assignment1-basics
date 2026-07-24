<!-- Possible working titles -->

# A twit idea

<!--

# Think like a git, work like a twit

# Twit or twig

# How do you keep worktrees together?

-->

Vibe coding has made worktrees a must!
But I'd found them useful even before I had a coding plan subscription
or could burn through my wallet by selecting Opus on Fast through OpenRouter.
I'm somewhat of a yak-shaving die-hard;
it's always tempting for me to turn a transient idea into a wild goose chase;
at least being almost expelled from college didn't help me stop doing it.
So: worktrees.
A gateway substance to let me feel
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
objects and refs are shared.
The directory and branch names need not match,
and Git has no opinion on where the directory should go.

That is about all the Git lesson this post needs.
The path argument was left to the caller.
My problem is with what callers have lately decided it means.

## Why do harness people hype about it?

For someone with a healthy life,
a second worktree lets the half-debugged thing remain half-debugged
while an urgent fix happens elsewhere.
The editor stays open, the breakpoints stay put,
and nobody has to remember whether the useful uncommitted change is in stash number three.

Agents did not invent [this use](https://github.blog/ai-and-ml/github-copilot/what-are-git-worktrees-and-why-should-i-use-them/);
but with token consumption being the best proxy indicator of welfare
that mankind can come up with, it's not at all surprising.
The harness has to put them somewhere.
[Claude Code](https://code.claude.com/docs/en/worktrees) uses `.claude/worktrees/`, and Codex uses `$CODEX_HOME/worktrees`;
the current [settings](https://learn.chatgpt.com/docs/environments/git-worktrees) at least let the user choose another root
and distinguish managed from permanent worktrees.
[`gwq`](https://github.com/d-kuro/gwq) builds a global `~/worktrees/host/owner/repo/branch` forest.
[Treehouse](https://github.com/kunchenguid/treehouse) leases warm numbered slots.
If the thing is disposable execution,
a cache or pool is exactly where it belongs.

[Worktrunk](https://github.com/max-sixty/worktrunk), the most visible specialist wrapper I found,
instead defaults to branch-derived siblings such as `repo.feature-auth`,
then adds cache copying, hooks, ports, agent launch and cleanup.
On Windows its `wt` collides with Windows Terminal,
so the package exposes `git-wt`;
`git wt` happens to work as an external Git command after all.
There are many more—`git gtr`, Branchlet, several Groves, several TUIs—and they do not converge
because a durable checkout, a throwaway chat, a reusable slot
and a peer around a bare controller are not the same product.

My unfinished tree is not a rented slot.
Treating it as one is the part I object to.

## But I'm not content with that...

What set me off was a child task
that had supposedly been sent to the worktree I made.
Codex quietly made another one under its own installation
and sent the task there instead.
Technically, splendid:
the files were separate.
Practically,
it had filed my work under product innards.

I want to see a worktree six weeks later
and know why I kept it.
I want to open it as an ordinary project,
move the whole household,
back up the odd note that belongs to the household rather than one checkout,
and not depend on the lifetime or current branding of an agent harness.
A chat ID is welcome to own a throwaway tree.
It does not get to own mine.

This also exposed a much older mistake in how I clone things.
Once `projects/somerepo` is the conventional clone,
it has already taken the clean name for the whole local project.
The second checkout becomes `projects/somerepo-auth`,
the third `projects/somerepo-parser`,
and soon the parent directory looks as if a repository burst.
Nesting them inside `somerepo/` makes one checkout pretend to be everybody's parent.
A global pool only moves the burst offstage.

The branch name is not much help with this.
The directory is where the abandoned thought physically lives;
the branch is one Git name associated with it.
I may want `auth/` in a window title
while the repository calls the branch `feat/oauth-retry`.
Git is already fine with that.

Nicholas Zakas calls direct siblings and a shared sibling container the two conventions
in his recent [introduction](https://humanwhocodes.com/blog/2026/07/introduction-git-worktrees/),
and prefers `my-project.worktrees/feature-name`.
I want the roof,
but I also want the original checkout beneath it.
Then the obvious basename means the whole local project,
not whichever checkout happened to arrive first.

The symmetrical version uses a bare repository as controller.
Morgan Cugerone described a clean
[bare worktree layout](https://morgan.cugerone.com/blog/how-to-use-git-worktree-and-in-a-clean-way/)
in 2021,
and this older
[layout note](https://gist.github.com/sellout/3361145fac9bf2dfdc6a9bc18dcdff36)
was already worrying the same bone.
This is cleaner than my answer:
the repository sits in the middle
and every working copy is honestly a peer.

Unfortunately I do not like it enough
to make every tool sit the entrance exam.
A [`--bare` clone](https://git-scm.com/docs/git-clone.html)
is not an ordinary clone with its files swept away:
among other differences,
remote heads are copied directly into local branch refs
instead of the usual `origin/*` arrangement.
IDEs, Git GUIs, hook installers, Git LFS, submodules,
language tools and miscellaneous project scripts
are routinely tested against an ordinary checkout.
Each bare-controller inconvenience is soluble.
Their sum is still a tax.

So:
a normal clone,
with the real `.git/` and a useful working tree,
demoted one directory
so it no longer monopolises the project's name.
The furniture can wait until the end.

## Break a leg with じゅじゅつ

[Jujutsu](https://docs.jj-vcs.dev/latest/) is where the extra roof stops being mere tidiness.
It calls its additional checkouts **workspaces**;
the glossary even defines one as
[what Git calls a worktree](https://docs.jj-vcs.dev/latest/glossary/#workspace).
Same idea, different paperwork.

A linked Git worktree has a `.git` pointer.
A secondary jj workspace has a `.jj/` directory
whose `repo` file points back to the initial jj repository.
Current jj refuses to initialise a colocated repository inside a linked Git worktree:
"Cannot create a colocated jj repo inside a Git worktree."
Going the other way is no cleverer:
`git status` in a jj-only secondary workspace finds no repository.
They can be neighbours,
but neither adopts the other's working copies.

The main checkout can bridge the repositories.
It may be a [colocated Git/jj repository](https://docs.jj-vcs.dev/latest/git-compatibility/),
with both `.git/` and `.jj/`.
Both systems then use the same Git object store,
and jj automatically imports and exports supported Git refs
when commands run there.
A Git-linked worktree and a jj secondary workspace can then be siblings of that common checkout.
They are still not each other's working copies.

The dotfiles reflect a real disagreement.
A Git worktree owns an index
and usually a branch or detached `HEAD`.
A jj workspace owns a working-copy commit,
shown as `workspace-name@`;
bookmarks are repository-wide names,
not "the current branch".
If another jj workspace changes that commit's tree,
the files can go stale
and jj asks for `jj workspace update-stale`.
Git's one-branch-per-worktree rule is guarding something else.

They even disagree about which half of deletion to perform:

```console
$ jj workspace add ../jj-second --name jj-second
$ jj workspace list
$ jj workspace forget jj-second
```

`forget` stops tracking the workspace
and leaves the directory.
Git's `worktree remove` removes the directory
and leaves the branch.

Tools such as [Treq](https://treq.dev/docs/concepts/workspaces/)
build a managed Git/jj hybrid above this.
The combination is useful.
The working copies are still not interchangeable.
This is the one case
where the neutral umbrella earns its rent without further pleading.

## See the grove and have the trees too

Next time, I am cloning into this:

```text
<projects>/
└── somerepo/                 # umbrella; normally just the repository basename
    ├── .prime/               # ordinary clone; main worktree; real .git/
    ├── dev/                  # a long-lived Git worktree
    ├── auth-retry/           # another Git worktree
    ├── parser-probe/         # perhaps temporary, but mine
    └── jj-type-inference/    # perhaps a jj workspace
```

The umbrella gets the repository basename.
If that has to coexist with an already conventional checkout,
`somerepo.grove/` is a tolerable variation.
When two remotes have the same basename,
`owner-somerepo/` or `host.owner.somerepo/` will do.
I do not intend to standardise the exceptions.

I see no point in a compulsory `worktrees/` layer beneath it.
The children already know what they are,
and so do `git worktree list` and `jj workspace list`.
If I am using both systems heavily,
`git/` and `jj/` subdirectories may become useful.
I will pay for that distinction
when I actually have it.

The base clone is an ordinary clone:

```powershell
New-Item -ItemType Directory somerepo | Out-Null
git clone REMOTE somerepo/.prime
Set-Location somerepo/.prime

git fetch origin
git worktree add -b feat/auth ../auth origin/main
git worktree list
```

Here `origin/main` stands for the remote's actual default branch.
An existing, unoccupied branch needs no new one:

```powershell
git worktree add ../auth feat/auth
```

The worktree directory is named for what I want in a shell prompt, editor title or task list.
It need not repeat the branch's namespace.
The branch remains the branch;
the directory is where I left the thing.

If I eventually tire of typing the path twice,
a small PowerShell function can derive it,
run the safety checks
and call Git's porcelain.
I do not need a template repository
for a convention that lives outside the repository.

For a new repository with no remote:

```powershell
New-Item -ItemType Directory -Force somerepo/.prime | Out-Null
git -C somerepo/.prime init -b main
# Make the first commit before adding linked worktrees.
```

If every Git that may touch the repository is 2.48 or newer,
relative worktree links make moving the whole umbrella less eventful:

```powershell
git config worktree.useRelativePaths true
```

That setting enables a repository extension
older Git refuses to open.
Absolute paths and a later `worktree repair` remain the compatibility-first choice.
I would not turn an existing dirty household into this layout with a clever one-liner:
preserve local refs, tracked changes, untracked files and ignored material first,
then either clone afresh
or move everything and repair all the linked paths.

The umbrella is not a repository.
Editors and coding agents should open `.prime/` or one of its siblings,
not the household.
Otherwise an agent given the "project root"
may quite reasonably regard every worktree as its territory.
The dot on `.prime` is a visual warning,
not a sandbox;
on Windows it does not even make the directory hidden.
Unix shells and plenty of file pickers do hide it,
so the price of the warning is having to open the base clone explicitly.

Repository scripts, hooks setup and `AGENTS.md` belong in the repository
and therefore arrive in every worktree.
A multi-root editor file or a note that really describes the whole household
may sit in the umbrella,
but Git cannot back up a file outside every repository.
I would keep that material scarce,
back it up separately,
and avoid an editor workspace that eagerly indexes every checkout
merely because they share a parent.

I am not turning this into six package-manager articles.
Share downloads;
do not share things being built or mutated.
Each Python worktree gets its own `.venv`
while uv shares its package cache.
The same line runs between a Julia depot and project,
Cargo's registry and `target/`,
a package-manager store and `node_modules`,
or compiler downloads and a CMake build directory.
Two build trees do not become safely concurrent
because their downloads were good neighbours.

The default hooks directory and ordinary Git configuration
are already shared through the common `.git`;
`core.hooksPath` may point somewhere else.
When a setting really belongs to one worktree,
enable `extensions.worktreeConfig`
and use `git config --worktree`;
don't infer the current checkout by scraping the `.git` file.
Git provides `git rev-parse --git-dir`, `--git-common-dir`, and `--git-path`
precisely because `.git` is not always a directory.
Enabling that extension is a repository-format choice:
older Git versions refuse to open it,
so it is not part of my bootstrap.

Removal is where a durable line of thought can accidentally be treated like a rented slot.
I want the boring ritual to enumerate tracked, untracked and ignored material
before the porcelain gets to decide anything:

```powershell
git -C ../auth status --short --branch --untracked-files=no
git -C ../auth ls-files --others --exclude-standard
git -C ../auth ls-files --others --ignored --exclude-standard
git worktree remove ../auth
git branch -d feat/auth
```

Ignored material is exactly where `.env`, virtual environments,
build trees and model checkpoints tend to live.
Removing the worktree and deleting the branch are separate acts;
`branch -d` gets its own chance to refuse.

If a directory was definitely deleted behind Git's back,
I can override the normal three-month expiry:

```powershell
git -C .prime worktree prune --dry-run --verbose --expire now
git -C .prime worktree prune --verbose --expire now
```

If the whole umbrella was moved behind Git's back,
give `repair` the new path of every linked worktree:

```powershell
git -C .prime worktree repair ../dev ../auth-retry ../parser-probe
```

`prune` removes stale administrative records,
not live worktree directories.
`repair` is for moved umbrellas and pointers.
Hand-editing `.prime/.git/worktrees/` is a fine way
to turn a directory convention into an archaeology project.
Nor should an eager cleanup script run [`git gc --prune=now`](https://git-scm.com/docs/git-gc.html)
while several agents may be writing objects;
Git's normal grace period exists for a reason.

Submodules remain the conspicuous exception.
Git's own [worktree manual](https://git-scm.com/docs/git-worktree#_bugs)
still calls their multiple-checkout support incomplete.
A linked worktree containing submodules cannot be moved with `git worktree move`,
and even a clean one needs `--force` to be removed.
I would not let an automatic janitor learn about that flag.

What should the main clone be called?
`root` is accurate
until an agent sees the word and gets ideas.
`anchor` and `control` sound like infrastructure products.
`repo` says nothing.
Repeating `somerepo/somerepo` is prosaic,
which is not a vice,
and gives editor windows a wonderfully normal title.

The dot-directory tournament was less dignified:
`.prime`, `.ansein`, `.noumenon`, `.self`, `.GIT`, `.twit`, `.canopy`, a repeated `.somerepo`,
`.master`, `.sith`, `.hermit`, `.inclave`, `.vader`, `.monos`, `.monad`,
`.demiurge`, `.init`, `.dictator`, `.stalin`,
and several things best left in the chat log.
`.host` was rejected.
`.seed` sounds like an RNG.
`.init` describes an event.
`.master` suggests the wrong relationship with the checked out branch.
`.demiurge` beats `.noumenon` in their particular niche.
Repeating the basename remains the option for people with better judgment.

I am choosing `.prime`.
Future me can reopen the tournament if he likes;
he will at least have to argue with a directory full of evidence.

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
- [Nicholas Zakas: An introduction to Git worktrees](https://humanwhocodes.com/blog/2026/07/introduction-git-worktrees/)
- [Morgan Cugerone: a bare-repository worktree layout](https://morgan.cugerone.com/blog/how-to-use-git-worktree-and-in-a-clean-way/)
- [Sellout's worktree layout note](https://gist.github.com/sellout/3361145fac9bf2dfdc6a9bc18dcdff36)
- [Git: `git-clone`](https://git-scm.com/docs/git-clone.html)
- [Jujutsu documentation](https://docs.jj-vcs.dev/latest/)
- [Jujutsu glossary: workspace](https://docs.jj-vcs.dev/latest/glossary/#workspace)
- [Jujutsu: colocated Git repositories](https://docs.jj-vcs.dev/latest/git-compatibility/)
- [Treq workspaces](https://treq.dev/docs/concepts/workspaces/)
- [Git: `git-gc`](https://git-scm.com/docs/git-gc.html)
- [Git worktree submodule caveat](https://git-scm.com/docs/git-worktree#_bugs)
