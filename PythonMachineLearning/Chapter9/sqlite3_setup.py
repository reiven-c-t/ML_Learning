import sqlite3
import os
conn = sqlite3.connect("reviews.sqlite3")
c = conn.cursor()
c.execute("CREATE TABLE review_db (review TEXT, sentiment INTEGER, date TEXT);")
example1 = "I love this movie"
c.execute("INSERT INTO review_db (review, sentiment, date) VALUES (?, ?, DATETIME('now'));", (example1, 1))
example2 = "I disliked this movie"
c.execute("INSERT INTO review_db (review, sentiment, date) VALUES (?, ?, DATETIME('now'));", (example2, 0))

c.execute("SELECT * FROM review_db WHERE date BETWEEN '2015-01-01 00:00:00' AND DATETIME('now');")
results = c.fetchall()
print(results)

conn.commit()
conn.close()
