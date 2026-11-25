# Moving the submission to Windows

Save the bundle (preferred) or ZIP and the supplied publish.ps1 into Downloads. Use the final commit and SHA-256 shown in the delivery receipt. The PowerShell script performs the local Git operations, stops on any failed check and requires an explicit -Publish switch to send changes. [PowerShell script](publish.ps1)

The bundle contains refs/heads/final-delivery and requires the existing base 792549e. The script verifies a clean tree and the fetch/push remote, switches to complete-unrippling, fetches origin, fast-forwards to the expected base, verifies and imports the bundle, checks both the commit ID and ancestry, and fast-forwards to the delivered commit. It checks the five-stage structure and report, then uses a normal push and verifies the remote ancestry.

For the ZIP fallback select -Method Zip and use the ZIP checksum. The ZIP is extracted to a new temporary directory. Its manifest lists added and modified paths, Git blob IDs, the intended Git tree and an empty removal list. Only those paths are copied into the clean checkout. No historical files or .git metadata are replaced. The script verifies the staged tree before making a new local commit. ZIP delivery reproduces the tree; bundle delivery also preserves the author's exact commit.

If local or remote history has advanced, stop and inspect `git log --graph --oneline --decorate --all` and `git diff 792549e..origin/complete-unrippling`. Import the delivery on a separate review branch, inspect both sets of changes, and reconcile them locally before publishing. Keep original changes and use normal fast-forward publication after review.
