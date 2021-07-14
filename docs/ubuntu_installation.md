## Ubuntu installation instructions (tested on 18.04) ##

### Install dependencies ###

```console
sudo apt install postgresql nodejs npm python3-pip apache2  apache2-dev libapache2-mod-wsgi-py3 libssl1.0-dev
```

### Configure PostgreSQL ###

* Login as postgres user: 
```console
sudo -i -u postgres
```

* Open postgres shell: 
```console
psql
```
* Open postgres shell: `psql`

* Create database and user:

```sql

CREATE DATABASE your_db_name;
CREATE USER your_user WITH PASSWORD ‘your_password’;
GRANT ALL PRIVILEGES ON DATABASE your_db_name TO your_user;
```

* Add extension for fulltext search:

```sql
\c your_db_name
CREATE EXTENSION pg_trgm;
\q
```

* Change local connections from peer to md5 in postgres config file (with vim or your prefered text editor: 
```console
sudo vim /etc/postgresql/10/pg_hba.conf
```

Fill in the database info in text-pair config: `sudo vim /etc/text-pair/config/global_settings.ini`

### Create webspace with proper permissions ###

```console
sudo mkdir /var/www/html/text-pair/
sudo chown -R $(whoami) /var/www/html/text-pair/
```

### Apache configuration (may require extra work) ###
```console
sudo a2enmod rewrite
sudo vim /etc/apache2/apache2.conf
```
Change AllowOverride from None to All

Add the following at the bottom of the file (to execute the wsgi search script):

`Include /etc/text-pair/*conf`

* Restart Apache: `sudo apachectl graceful`

#### NOTE ####
If you get a syntax error for the Include directive in Apache, replace that line with:
`IncludeOptional /etc/text-pair/*conf`

### Run install script ###
Run the following script at the root of the text-pair folder:

`sh install.sh`
