
### How to run:
This is expected to be run inside the `dev-client` service as specified in `./cluster/docker-compose.yml`. One way to do this after starting `docker-compose up -d --build` in the `cluster` directory, use `Dev Containers` vscode extension to connect to the dev-client container (right click `cluster-dev-client` and select attach visual studio code) and open the `/home/jovyan/workspace` folder where this file will be and run it there (may need to install some extensions for ease of running and debugging).

You may want to modify docker defaults to allow for the memory listed in the `./cluster/docker-compose.yml`.

Be sure to run `docker-compose down` in the `./cluster/` to remove these resources after you are done.