from pymongo import MongoClient

# Connect to the MongoDB server
client = MongoClient("mongodb+srv://abhiram:abhiram123@cluster0.rrv8gqp.mongodb.net/")

# Access the database
db = client['interview']  # Replace 'your_database_name' with your actual database name

# Create a collection (similar to a table in SQL)
collection = db['users']  # Replace 'your_collection_name' with your actual collection name

# Now you can insert documents (rows) into the collection
data = {
    'name': 'John Doe',
    'email': 'john.doe@example.com',
    'username': 'johndoe123',
    'password': 'secret'
}
result = self.collection.insert_one(data)
print("Document inserted successfully")
