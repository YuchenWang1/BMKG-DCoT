## data
提供了一个处理流程，用于将 Neo4j 导出的原始图数据转换为更容易检索和分析的结构化格式。工作流程包括原始数据（`raw.json`）、处理脚本（`processing.py`），以及多个用于查询和测试的 JSON 文件。

### 文件夹内容
- **raw.json**  
  从 Neo4j 导出的原始图数据。  

- **processing.py**  
  Python 脚本，将 `raw.json` 转换为结构化、可检索的图文件 `graph.json`。  

- **graph.json**  
  由 `processing.py` 脚本生成的处理后图数据。该格式根据节点类型组织，便于高效检索。  

- **data.json**  
  查询文件，包含标准的检索语句。  

- **data_missing.json**  
  查询文件，包含模糊或不完整的查询。可用于测试系统对缺失信息的鲁棒性。  

- **cache-all-mpnet-base-v2.pkl**  
  缓存的模型文件（用于向量嵌入/语义检索）。  

### 使用方法
1. 将 `raw.json` 放入项目文件夹中。  
2. 运行转换脚本：  
   ```bash
   python processing.py
   ```  
3. 会生成输出文件 `graph.json`。  
4. 使用 `data.json` 和 `data_missing.json` 进行图检索测试。  
