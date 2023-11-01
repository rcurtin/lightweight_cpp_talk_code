FROM openjdk:22-slim

ADD hello_world.scala .

RUN apt-get -yy update && \
    apt-get -yy install scala && \
    rm -rf /var/cache/apt/* && \
    scalac hello_world.scala

ENTRYPOINT scala hello
