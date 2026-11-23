---
title: Docker
parent: Installation
nav_order: 1
---

# Docker

The recommended install is to build your own Docker image and run TextPAIR inside a container.

1. From the root of the repository, build the image:

   ```console
   docker build -t textpair .
   ```

2. Start a container:

   ```console
   docker run -td -p 80:80 --name textpair textpair init_textpair_db
   ```

   You may want to customize the `run` command to your needs, for instance to mount a volume holding your data.

3. Copy your texts into the container, then follow [Running an alignment](running.html) from inside it.

If the web server does not respond, restart it inside the container:

```console
/var/lib/text-pair/api_server/web_server.sh &
```
