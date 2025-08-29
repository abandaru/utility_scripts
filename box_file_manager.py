import os
import json
import time
from pathlib import Path
from boxsdk import Client, OAuth2, JWTAuth
from boxsdk.exception import BoxAPIException

class BoxFileManager:
    def __init__(self):
        self.client = None
        self.config_file = "box_config.json"
        
    def authenticate_oauth2(self, client_id, client_secret):
        """
        Authenticate using OAuth2 (requires user interaction)
        You'll need to register an app at https://developer.box.com/
        """
        try:
            auth = OAuth2(
                client_id=client_id,
                client_secret=client_secret,
                store_tokens=self._store_tokens
            )
            
            # Get authorization URL
            auth_url, csrf_token = auth.get_authorization_url('http://localhost')
            print(f"\nPlease visit this URL to authorize the application:")
            print(f"{auth_url}")
            
            # Get authorization code from user
            auth_code = input("\nEnter the authorization code from the redirect URL: ")
            
            # Exchange auth code for tokens
            access_token, refresh_token = auth.authenticate(auth_code)
            
            self.client = Client(auth)
            print("Successfully authenticated with OAuth2!")
            return True
            
        except Exception as e:
            print(f"OAuth2 authentication failed: {e}")
            return False
    
    def authenticate_jwt(self, config_path):
        """
        Authenticate using JWT (Service Account)
        Requires a JSON config file from Box Developer Console
        """
        try:
            auth = JWTAuth.from_settings_file(config_path)
            self.client = Client(auth)
            
            # Test the connection
            user = self.client.user().get()
            print(f"Successfully authenticated as: {user.name}")
            return True
            
        except Exception as e:
            print(f"JWT authentication failed: {e}")
            return False
    
    def _store_tokens(self, access_token, refresh_token):
        """Store tokens for future use"""
        tokens = {
            'access_token': access_token,
            'refresh_token': refresh_token
        }
        with open(self.config_file, 'w') as f:
            json.dump(tokens, f)
    
    def list_files(self, folder_id='0', limit=100):
        """List files in a folder (0 = root folder)"""
        if not self.client:
            print("Not authenticated. Please authenticate first.")
            return []
        
        try:
            folder = self.client.folder(folder_id).get()
            items = folder.get_items(limit=limit)
            
            files = []
            print(f"\nFiles in folder '{folder.name}':")
            print("-" * 50)
            
            for item in items:
                if item.type == 'file':
                    files.append({
                        'id': item.id,
                        'name': item.name,
                        'size': item.size,
                        'modified': item.modified_at
                    })
                    print(f"File: {item.name} (ID: {item.id})")
                elif item.type == 'folder':
                    print(f"Folder: {item.name} (ID: {item.id})")
            
            return files
            
        except BoxAPIException as e:
            print(f"Error listing files: {e}")
            return []
    
    def download_file(self, file_id, local_path=None):
        """Download a file from Box"""
        if not self.client:
            print("Not authenticated. Please authenticate first.")
            return False
        
        try:
            file_obj = self.client.file(file_id).get()
            
            if local_path is None:
                local_path = file_obj.name
            
            # Create directory if it doesn't exist
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            
            print(f"Downloading '{file_obj.name}' to '{local_path}'...")
            
            with open(local_path, 'wb') as f:
                file_obj.download_to(f)
            
            print(f"Successfully downloaded: {local_path}")
            return local_path
            
        except BoxAPIException as e:
            print(f"Error downloading file: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error: {e}")
            return False
    
    def upload_file(self, local_path, folder_id='0', file_id=None):
        """
        Upload a file to Box
        If file_id is provided, it will create a new version of the existing file
        """
        if not self.client:
            print("Not authenticated. Please authenticate first.")
            return False
        
        try:
            if not os.path.exists(local_path):
                print(f"Local file not found: {local_path}")
                return False
            
            file_name = os.path.basename(local_path)
            
            if file_id:
                # Upload new version of existing file
                print(f"Uploading new version of '{file_name}'...")
                file_obj = self.client.file(file_id)
                updated_file = file_obj.update_contents(local_path)
                print(f"Successfully uploaded new version: {updated_file.name} (Version: {updated_file.version_number})")
                return updated_file
            else:
                # Upload new file
                print(f"Uploading new file '{file_name}' to folder {folder_id}...")
                folder = self.client.folder(folder_id)
                uploaded_file = folder.upload(local_path, file_name)
                print(f"Successfully uploaded: {uploaded_file.name} (ID: {uploaded_file.id})")
                return uploaded_file
                
        except BoxAPIException as e:
            print(f"Error uploading file: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error: {e}")
            return False
    
    def search_files(self, query, limit=10):
        """Search for files by name"""
        if not self.client:
            print("Not authenticated. Please authenticate first.")
            return []
        
        try:
            results = self.client.search().query(
                query=query,
                limit=limit,
                file_extensions=['txt', 'pdf', 'docx', 'xlsx', 'csv']  # Adjust as needed
            )
            
            files = []
            print(f"\nSearch results for '{query}':")
            print("-" * 50)
            
            for item in results:
                if item.type == 'file':
                    files.append({
                        'id': item.id,
                        'name': item.name,
                        'path': item.path_collection['entries'] if hasattr(item, 'path_collection') else []
                    })
                    print(f"Found: {item.name} (ID: {item.id})")
            
            return files
            
        except BoxAPIException as e:
            print(f"Error searching files: {e}")
            return []

def main():
    manager = BoxFileManager()
    
    print("Box Drive File Manager")
    print("=" * 40)
    
    # Authentication options
    print("\nAuthentication Options:")
    print("1. OAuth2 (requires client ID and secret)")
    print("2. JWT (requires JSON config file)")
    
    auth_choice = input("\nChoose authentication method (1 or 2): ")
    
    if auth_choice == '1':
        client_id = input("Enter your Box App Client ID: ")
        client_secret = input("Enter your Box App Client Secret: ")
        
        if not manager.authenticate_oauth2(client_id, client_secret):
            print("Authentication failed. Exiting.")
            return
            
    elif auth_choice == '2':
        config_path = input("Enter path to JWT config file (or press Enter for 'box_config.json'): ")
        if not config_path:
            config_path = 'box_config.json'
        
        if not manager.authenticate_jwt(config_path):
            print("Authentication failed. Exiting.")
            return
    else:
        print("Invalid choice. Exiting.")
        return
    
    while True:
        print("\n" + "=" * 40)
        print("Available Actions:")
        print("1. List files in root folder")
        print("2. Search for files")
        print("3. Download file")
        print("4. Upload new file")
        print("5. Update existing file")
        print("6. Exit")
        
        choice = input("\nSelect an action (1-6): ")
        
        if choice == '1':
            manager.list_files()
            
        elif choice == '2':
            query = input("Enter search term: ")
            manager.search_files(query)
            
        elif choice == '3':
            file_id = input("Enter file ID to download: ")
            local_path = input("Enter local path (or press Enter for default): ")
            if not local_path:
                local_path = None
            
            downloaded_path = manager.download_file(file_id, local_path)
            if downloaded_path:
                print(f"\nFile downloaded to: {downloaded_path}")
                print("You can now edit the file locally.")
                input("Press Enter when you're done editing...")
                
                # Ask if user wants to upload the updated file
                upload_choice = input("Upload the updated file back to Box? (y/n): ")
                if upload_choice.lower() == 'y':
                    manager.upload_file(downloaded_path, file_id=file_id)
        
        elif choice == '4':
            local_path = input("Enter path to local file: ")
            folder_id = input("Enter folder ID (or press Enter for root): ")
            if not folder_id:
                folder_id = '0'
            
            manager.upload_file(local_path, folder_id)
            
        elif choice == '5':
            file_id = input("Enter file ID to update: ")
            local_path = input("Enter path to updated file: ")
            
            manager.upload_file(local_path, file_id=file_id)
            
        elif choice == '6':
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
