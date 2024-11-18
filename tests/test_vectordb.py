import uuid


def test_vectordb_create_collection(vector_db_client):
    vector_db_client.create_collection("test_collection")
    assert vector_db_client.collection_exists("test_collection")


def test_vectordb_create_vector(vector_db_client):
    vector_db_client.create_collection("test_collection")
    vector_db_client.add_points(
        "test_collection",
        [
            {"id": 1, "text": "hello world", "role": "greeting"},
            {"id": str(uuid.uuid4()), "text": "goodbye world", "role": "farewell"},
        ],
    )
