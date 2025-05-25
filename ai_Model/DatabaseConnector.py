import psycopg2
import os
from jproperties import Properties

class DatabaseConnector:
    def __init__(self, properties_filename="application.properties"):
        # Get current script directory dynamically
        script_directory = os.path.dirname(os.path.realpath(__file__))
        properties_path = os.path.join(script_directory, properties_filename)

        if not os.path.exists(properties_path):
            raise FileNotFoundError(f"Properties file does not exist: {properties_path}")

        # Load the properties file
        configs = Properties()
        with open(properties_path, "rb") as config_file:
            configs.load(config_file)

        # Fetch database connection parameters
        db_url = configs.get("spring.datasource.url").data
        self.user = configs.get("spring.datasource.username").data
        self.password = configs.get("spring.datasource.password").data

        # Parse db_url to get host, port, dbname
    
        if db_url.startswith("jdbc:postgresql://"):
            db_url = db_url.replace("jdbc:postgresql://", "")
        host_port_db = db_url.split(":")
        self.host = host_port_db[0]
        port_dbname = host_port_db[1].split("/")
        self.port = port_dbname[0]
        self.dbname = port_dbname[1]

        print(f"DatabaseConnector initialized: host={self.host}, port={self.port}, dbname={self.dbname}, user={self.user}")

    def connect(self):
        try:
            conn = psycopg2.connect(
                dbname=self.dbname,
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port
            )
            return conn
        except Exception as e:
            print(f"Error connecting to database: {e}")
            return None

    def insert_conversation(self, name, input, output= "default"):
        conn = self.connect()
        if conn:
            try:
                cursor = conn.cursor()
                insert_query = """
                INSERT INTO data (name, input, output)
                VALUES (%s, %s, %s);
                """
                cursor.execute(insert_query, (name, input, output))
                conn.commit()
                print("Conversation saved successfully.")
            except Exception as e:
                print(f"Error inserting into database: {e}")
            finally:
                cursor.close()
                conn.close()
    
    def update_column_types(self):
        conn = self.connect()
        if conn:
            try:
                cursor = conn.cursor()

                # Alter columns to TEXT type
                alter_queries = [
                 
                    "ALTER TABLE data ALTER COLUMN input TYPE TEXT;",
                    "ALTER TABLE data ALTER COLUMN output TYPE TEXT;"
                ]
                
                for query in alter_queries:
                    cursor.execute(query)

                # Commit the changes
                conn.commit()
                print("Columns updated successfully.")
            except Exception as e:
                print(f"Error updating columns: {e}")
            finally:
                cursor.close()
                conn.close()


    def fetch_history(self):
        conn = self.connect()
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT name, input, output FROM data;")
                rows = cursor.fetchall()
                return [{"name": row[0], "input": row[1], "output": row[2]} for row in rows]  # Return data as a list of dicts
            except Exception as e:
                print(f"Error fetching history: {e}")
                return []
            finally:
                cursor.close()
                conn.close()
        return []   
    




# detta är nytt 


    def create_preferences_table(self):
        """Skapa en tabell för användarpreferenser om den inte finns."""
        conn = self.connect()
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                id SERIAL PRIMARY KEY,
                name TEXT UNIQUE,
                text TEXT,
                user_info TEXT DEFAULT 'default'
                );
                """)
                conn.commit()
            except Exception as e:
                print(f"Error creating preferences table: {e}")
            finally:
                cursor.close()
                conn.close()

    def insert_user_preference(self, name, text, user_info="default"):
        """Lägg till eller uppdatera en användarpreferens."""
        conn = self.connect()
        if conn:
            try:
                cursor = conn.cursor()
                insert_query = """
                INSERT INTO user_preferences (name, text, user_info)
                VALUES (%s, %s, %s)
                ON CONFLICT(name) DO UPDATE SET text = EXCLUDED.text, user_info = EXCLUDED.user_info;
                """
                cursor.execute(insert_query, (name, text, user_info))
                conn.commit()
                print(f"✅ Preference '{name}' saved successfully.")
            except Exception as e:
                print(f"❌ Error inserting preference: {e}")
            finally:
                cursor.close()
                conn.close()


    def fetch_user_preferences(self):
        """Hämta alla användarpreferenser."""
        conn = self.connect()
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT name, text, user_info  FROM user_preferences;")
                rows = cursor.fetchall()
                return [{"name": row[0], "text": row[1], "user_info": row[2]} for row in rows] # Returnera preferenser som en dictionary
            except Exception as e:
                print(f"Error fetching preferences: {e}")
                return {}
            finally:
                cursor.close()
                conn.close()
        return {}