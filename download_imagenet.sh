#!/bin/sh
cat list/urllist.txt | xargs -n 2 ./process_urls.sh > download.log 2> error.log
