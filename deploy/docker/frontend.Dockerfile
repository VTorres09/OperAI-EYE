FROM nginx:1.29-alpine

COPY deploy/docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY edge_app/ui/ /usr/share/nginx/html/

EXPOSE 80
