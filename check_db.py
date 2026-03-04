import sqlite3

# Connect to the appointments database
conn = sqlite3.connect("appointments.db")
cursor = conn.cursor()

# Fetch all rows from the appointments table
cursor.execute("SELECT * FROM appointments")
rows = cursor.fetchall()

# Check if data exists
if rows:
    print("Appointments found in the database:\n")
    for row in rows:
        print(row)
else:
    print("No appointments found.")

conn.close()
