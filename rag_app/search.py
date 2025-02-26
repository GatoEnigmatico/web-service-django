from elasticsearch_dsl import Document, Text, connections

# Asegúrate de usar la URL completa con esquema y puerto
connections.create_connection(
    hosts=['http://localhost:9200'],
    verify_certs=False
)

class DocumentIndex(Document):
    title = Text()
    content = Text()

    class Index:
        name = 'documents_index'

    def save(self, **kwargs):
        return super().save(**kwargs)