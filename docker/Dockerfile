FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y postgresql postgresql-contrib curl git locales libpq-dev python3-pip sudo

RUN curl -sL https://deb.nodesource.com/setup_14.x | sudo -E bash && apt-get install -y nodejs

RUN apt-get clean && rm -rf /var/lib/apt

RUN mkdir textpair && curl -L  https://github.com/ARTFL-Project/text-pair/archive/v2.0-beta.5.tar.gz | tar xz -C textpair --strip-components 1 &&\
    cd textpair && sh install.sh

RUN echo "[WEB_APP]\nweb_app_path = /var/www/html/text-pair\n[DATABASE]\ndatabase_name = textpair\ndatabase_user = textpair\ndatabase_password = textpair" > /etc/text-pair/global_settings.ini

# Set the locale
RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && \
    locale-gen
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

COPY docker_autostart.sh /usr/local/bin/init_textpair_db

CMD ["/usr/local/bin/init_textpair_db"]
