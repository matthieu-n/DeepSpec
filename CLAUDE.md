# Agent guidance for this repo

## GitHub account for pushing

`origin` (`matthieu-n/DeepSpec`) is a personal-account fork. The default
`gh`/git credential is `matthieu-neau_ddog`, a GitHub EMU account —
EMU accounts cannot push to (or fork) external public repos like this
one; pushes will 403.

Before any `git push`, switch the active `gh` account:

```
gh auth switch --hostname github.com --user matthieu-n
gh auth setup-git
```

Switch back to `matthieu-neau_ddog` afterward if working in dd-source
in the same terminal session.

## Upstream

`upstream` remote points at `deepseek-ai/DeepSpec` (the repo this was
forked from). Pull updates with:

```
git fetch upstream
git merge upstream/main
```
