FROM alpine:latest

ADD hello_world.cpp .

RUN apk add g++ && \
    g++ -o hello_world hello_world.cpp && \
    apk del --purge g++ && \
    rm -rf /var/cache/ && \
    rm hello_world.cpp

ENTRYPOINT ./hello_world
