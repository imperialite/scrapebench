# robots.txt for http://www.gymglish.com/
# More on http://www.robotstxt.org
sitemap: https://www.gymglish.com/sitemapindex.xml
User-agent: *

# Exclude some URLs
Disallow: /wb/www/notretemps-www/
Disallow: /nosquid/
Disallow: /www/
Disallow: /virtual/
#Disallow: /partner/ -> handled using meta tags
Disallow: */cpartner/
Disallow: */partner/
Disallow: */lps/
Disallow: /api/

# Allow online shop, but disallow rest of workbook
Allow: /workbook/shop
Disallow: /workbook/
Disallow: */workbook/
Disallow: */showlesson
Disallow: /backoffice/

# Disallow media
Disallow: /audios/
Disallow: /videos/
Disallow: /opensource/

# Referral links & registration pages
Disallow: /r/

# Exclude some documents
Disallow: /documents/newsletter/
Disallow: /documents/CP/
Disallow: /documents/gymglish_examples/
