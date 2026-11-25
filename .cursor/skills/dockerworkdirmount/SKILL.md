---
name: dockerworkdirmount
description: After `bash script/docker.sh` the container shell starts in an EMPTY directory owned by root -- no repo files, `touch: Permission denied`, nothing works. The checkout (or cwd) lives outside the mounted /proj and /scratch, e.g. /storage1.
---

# docker.sh: empty, unwritable working directory

## Symptom
`bash script/docker.sh` opens a shell, `pwd` prints the checkout path, but
`ls -la` shows an empty `root root` directory and any write fails with
`Permission denied`.

## Cause
`docker run -w <dir>` on a path that is **not** a bind mount makes Docker
create that directory inside the container, owned by root and empty. The
container runs as your own uid (the Dockerfile's `USER`), so it can neither
see the repo nor write. docker.sh used to mount only `/proj` and `/scratch`,
so any checkout elsewhere (`/storage1/...`, `$HOME/...`) hit this.

## Fix (in script/docker.sh)
The script maps 1:1 (rslave) `/proj`, `/scratch`, the parent of the checkout
(e.g. `/storage1/staff/<user>`), the parent of the cwd, and the checkout and
cwd themselves -- shortest path first, skipping any folder already inside a
mapped one, never `/`. Other host paths go in `AIEHLC_DOCKER_MOUNTS="/a /b"`.
An old long-lived container (`AIEHLC_DOCKER_DETACH=1`) keeps its old mounts:
`docker rm -f <name>` and start it again.

## Check
    docker run --rm -w "$(pwd)" aiehlc-dev:latest ls -la   # old behaviour: empty, root-owned
Inside the new container: `ls` shows the repo and `touch x && rm x` works.
