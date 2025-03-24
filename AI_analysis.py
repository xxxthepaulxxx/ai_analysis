import pandas as pd
import os,sys
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, BulkWriteError
from sentence_transformers import SentenceTransformer
from pymongo.operations import SearchIndexModel
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

# sentences = ["This is an example sentence", "Each sentence is converted"]
path = os.getcwd().replace("\\","/")
print (path)
# model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True ) ##dimension: 768
# model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2') #dimension: 384
# model = SentenceTransformer('all-MiniLM-L6-v2')
model_save_path = f'{path}/models/all-MiniLM-L6-v2'


#model from local
def load_model(model_save_path):
    model = SentenceTransformer(model_save_path)
    return model

# Save the model to local directory

def save_model(model,model_save_path):
    if not os.path.exists(model_save_path):
        print(f"Saving model to {model_save_path}")
        model.save(model_save_path)
        print("Model saved successfully")
    else:
        print(f"Model already exists at {model_save_path}")
        # Load the model from local path
        model = SentenceTransformer(model_save_path)
    return model

# embeddings = model.encode(sentences)
def get_embedding(data):
    """Generates vector embeddings for the given data."""
    embedding = model.encode(data)
    return embedding.tolist()

def import_tags_to_mongodb(excel_file_path, sheet_name=None):
    """
    Imports tags and descriptions from an Excel file into MongoDB Atlas.

    Parameters:
    - excel_file_path (str): Path to the Excel file.
    - sheet_name (str, optional): Name of the sheet to read. If None, reads the first sheet.

    Returns:
    - None
    """
    try:
        if sheet_name:
            df = pd.read_excel(excel_file_path, sheet_name=sheet_name)
        else:
            df = pd.read_excel(excel_file_path)

        if 'name' not in df.columns or 'descriptions' not in df.columns:
            raise ValueError("Excel file must contain 'name' and 'descriptions' columns.")

        documents = []
        for index, row in df.iterrows():
            print(f"Ingesting TAG:  {row['name']} ... ")
            tag_name = row['name']
            tag_description = row['descriptions']
            name_description = tag_name + tag_description
            name_embeded = get_embedding(tag_name)
            description_embeded = get_embedding(tag_description)
            name_description_embeded = get_embedding(name_description)
            
            if pd.isna(tag_name) or pd.isna(tag_description):
                print(f"Skipping row {index + 2} due to missing data.")
                continue

            document = {
                "name": str(tag_name).strip(),
                "descriptions": str(tag_description).strip(),
                "name_embeded" : name_embeded,
                "description_embeded" : description_embeded,
                "name_description_embeded" : name_description_embeded
            }
            documents.append(document)

        if not documents:
            print("No valid documents to insert.")
            return

        
        try:
            client.admin.command('ping')
            print("Successfully connected to MongoDB Atlas.")
        except ConnectionFailure:
            print("Failed to connect to MongoDB Atlas. Check your connection string and network settings.")
            return


        try:
            result = collection.insert_many(documents, ordered=False)
            print(f"Successfully inserted {len(result.inserted_ids)} documents into 'recommendation_system' collection.")
        
        except BulkWriteError as bwe:
            print("Bulk write error occurred:", bwe.details)
        except Exception as e:
            print("An error occurred while inserting documents:", str(e))

    except FileNotFoundError:
        print(f"The file {excel_file_path} was not found.")
    except ValueError as ve:
        print("ValueError:", ve)
    except Exception as ex:
        print("An unexpected error occurred:", str(ex))

def query_by_vector_search(query,path,search_index):

    result_embeded = get_embedding(query)
    # print(f"EMBEDED RESULT: {result_embeded}")
    results = collection.aggregate([
    {"$vectorSearch": {
        "queryVector": result_embeded,
        "path": path, 
        "numCandidates": 100, 
        "limit": 5, #top 4 matching
        "index": search_index,
        }},
    {"$addFields": {"similarity_score": {"$meta": "vectorSearchScore"}}}
    ])
    for document in results:
        print(f'TAG : {document["name"]},\nTAG_NAME: {document["descriptions"]},\nScore: {document["similarity_score"]}\n')
    return results

# #create vector-search-index
def create_vector_index(path,search_index_col,num_dimensions,collection):
    search_index_model = SearchIndexModel(
    definition={
    "fields": [
    {
        "type": "vector",
        "path": path, 
        "numDimensions": num_dimensions,   # 384/ nomic with 768
        "similarity": "dotProduct"
    }
    ],
    },
    name=search_index_col,
    type="vectorSearch",
    )   
    result = collection.create_search_index(model=search_index_model)
    print("New search index named " + result + " is building.")

def perform_field_clustering(collection, field_path, n_clusters=5):
    """
    Perform clustering on specified field embeddings from MongoDB collection.
    
    Parameters:
    - collection: MongoDB collection
    - field_path: The field containing embeddings (e.g., 'name_embeded', 'description_embeded')
    - n_clusters: Number of clusters to create
    
    Returns:
    - Dictionary mapping document IDs to cluster labels
    """
    # Fetch all documents with the specified embedding field
    documents = list(collection.find({}, {'_id': 1, 'name': 1, field_path: 1}))
    
    if not documents:
        print("No documents found in collection")
        return None
    
    # Extract embeddings and document info
    embeddings = []
    doc_info = []
    
    for doc in documents:
        if field_path in doc:
            embeddings.append(doc[field_path])
            doc_info.append({'id': doc['_id'], 'name': doc['name']})
    
    if not embeddings:
        print(f"No embeddings found in field '{field_path}'")
        return None
    
    # Convert to numpy array
    embeddings_array = np.array(embeddings)
    
    # Standardize the features
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embeddings_array)
    print(f"Scaled embeddings shape: {scaled_embeddings.shape}")
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    print(f"KMeans clustering with {n_clusters} clusters")
    cluster_labels = kmeans.fit_predict(scaled_embeddings)
    
    # Create cluster mapping and update documents
    for doc, cluster_label in zip(doc_info, cluster_labels):
        print(f"Field: {doc['name']}, Cluster: {cluster_label}")
        # Update the document in MongoDB with its cluster
        collection.update_one(
            {'_id': doc['id']},
            {'$set': {'cluster': int(cluster_label)}}
        )
    
    # Count documents in each cluster
    cluster_counts = np.bincount(cluster_labels)
    for cluster_idx, count in enumerate(cluster_counts):
        print(f"\nCluster {cluster_idx}: {count} documents")
    
    return dict(zip([doc['id'] for doc in doc_info], cluster_labels))

def main_import():
    # 匯入資料
    # import data from csv to assigned collection
    import_tags_to_mongodb(excel_file, sheet)
    # 建立vector_index
    path = "name_description_embeded"
    search_index_col = "vector_index_desc"
    num_dimensions = 384
    create_vector_index(path,search_index_col,num_dimensions,collection)

def main_vector_search():
    
    # MAKE A REQUEST
    # query = ["吃飯皇帝大我要全部最佳信用卡_基金愛好","喔是名稱"]
    query = "吃飯皇帝大我要全部最佳信用卡_基金愛好"
    # query = "喔是名稱"

    serch_path = "name_description_embeded" # description_embeded / name_embeded
    search_index = "vector_index_desc" # vector_index_desc/vector_index_name
    answer = query_by_vector_search(query,serch_path,search_index)


def main_cluster(field_path,n_clusters):
    # Example usage of the clustering function
    cluster_results = perform_field_clustering(collection, field_path, n_clusters)    

if __name__ == "__main__":
    excel_file = "tag.xlsx" 
    sheet = None 
    # mongo_uri = "mongodb+srv://justinyu1101:kF9FmIEe8t7NSOA7@cluster0.0lrcr.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    mongo_uri = "mongodb+srv://xxxthepaulxxx:abcd1234@cluster0.8pj7i.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    db = client['recommendation_system']
    collection = db['tag_test'] #tag_all_nomic / tag_all / tag_all_miniLM
    
    # save_model(model,model_save_path)
    # #create vector-search-index 

    # AI RECOMMENDATION 
    ## import data and create vector-search-index
    # main_import()
    # 使用本地模型
    model_local_path = f'{path}/models/all-MiniLM-L6-v2'
    model = load_model(model_local_path)    
    # 使用遠端模型
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2') #dimension: 384
    
    ## 測試 tag recommendation
    main_vector_search()


    ## 測試 clustering
    # field_path = "name_description_embeded"  # or "description_embeded" or "name_description_embeded"
    # n_clusters = 10
    # # main_import()
    # # main_vector_search()
    # main_cluster(field_path,n_clusters)
