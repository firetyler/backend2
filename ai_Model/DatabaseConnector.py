from jproperties import Properties
import psycopg2
import os

class DatabaseConnector:
    def __init__(self, properties_path="application.properties"):
        # Check if properties file exists
        if not os.path.exists(properties_path):
            print(f"Error: The properties file does not exist at: {properties_path}")
            return

        # Load the application.properties file
        configs = Properties()
        with open(properties_path, "rb") as config_file:
            configs.load(config_file)

        # Read properties
        db_url = configs.get("spring.datasource.url").data
        self.user = configs.get("spring.datasource.username").data
        self.password = configs.get("spring.datasource.password").data

        # Parse database URL (example: jdbc:postgresql://localhost:5432/mydatabase)
        db_host_port = db_url.split("//")[1]
        self.host, port_dbname = db_host_port.split(":")
        self.port, self.dbname = port_dbname.split("/")

    def connect(self):
        try:
        # Return psycopg2 connection object
            conn = psycopg2.connect(
            dbname=self.dbname,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port
        )
            print("Database connected successfully!")
            return conn
        except Exception as e:
            print(f"Error connecting to database: {e}")
            return None

# Example usage
if __name__ == "__main__":
    # Adjust the path to the correct location of your application.properties
    db = DatabaseConnector(properties_path="C:/Users/olive/Documents/java/backend/backend/src/main/resources/application.properties")
    if db:  # Check if db is initialized correctly
        conn = db.connect()

        print("Connected to database!")

        # Always remember to close when done
        conn.close()
        print("Connection closed.")
