# pull official base image
FROM node:16-alpine

# set working directory
WORKDIR /web

# add `/web/node_modules/.bin` to $PATH
ENV PATH /web/node_modules/.bin:$PATH

# install web dependencies
COPY package.json ./
COPY yarn.lock ./

# wait for some dependency
ADD https://github.com/ufoscout/docker-compose-wait/releases/download/2.2.1/wait /wait
RUN chmod +x /wait

# start web
CMD /wait && yarn start ; if [ $? -ne 0 ] ; then yarn install && yarn start ; fi
