# OD_Project

Research Project of Open Data Course: Proof of Concept - Graph Embedding

## Proof of Concept

### Graph embedding using Neo4J

Two distinct data sources were evaluated in the project. The first one consists of the dblp data source. This data provides bibliographic information of the main computer science journals and proceedings and it was adopted in the first project of the Open Data course. This graph models and store the connections between author, publications and journals/proceedings. 

Data obtained from: https://dblp.org/xml/release/dblp-2019-11-01.xml.gz

![graph_dblp](https://user-images.githubusercontent.com/79153695/123142883-38cc7280-d45a-11eb-913b-699afaec84f1.png)

The second data source is the Neo4j twitter data. This data consist of sample of the data from Neo4J's personal Twitter account. It includes activities of the Neo4J account, plus interactions and citations of users with this account.

![twitter](https://user-images.githubusercontent.com/79153695/123143080-703b1f00-d45a-11eb-8270-0211e2a2deb9.png)

Data obtained from: https://github.com/neo4j-graph-examples/twitter-v2

### Execution

Neo4j version: 4.2.5

Load graph data via the following:

Dump file data set 1: neo4j_dump/dblp_data_link.txt     -> Google drive link (160MB)
Dump file data set  2: neo4j_dump/twitter-v2-40.dump

* Drop the file into the Files section of a project in Neo4j Desktop. Then choose the option to Create new DBMS from dump option from the file options.

Ptyhon version: 3.9

Execute code after loading the graphs in Neo4J.

Code data set 1: code/embed_ML_dblp.py
Code data set 2: code/embed_ML_twitter.py

Python requirements file: code/requirements.txt

Before execution, line 9 of the codes must be updated to include local drive of neo4j:

* driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "teste"))
