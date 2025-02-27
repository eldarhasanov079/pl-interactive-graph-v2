# An Interactive Tool for Randomized Autogradable Graph Assessments

## Overview  
This project introduces an **interactive graph-based assessment tool** designed for online learning platforms. It allows students to dynamically interact with graph models by clicking on nodes and edges to simulate different algorithms. The tool supports a range of graph-based problems, from **basic traversals (BFS, DFS)** to **advanced algorithms (Dijkstra’s, Kruskal’s, Hypercube processing)**.

📢 **Presented at [SIGCSE 2025 Technical Symposium](https://sigcse2025.org/) in Pittsburgh, PA.**  
📄 **Publication: [An Interactive Tool for Randomized Autogradable Graph Assessments](https://dl.acm.org/doi/10.1145/3641555.3705123)** 

## Features  
- 🎯 **Interactive Graph Manipulation** – Click nodes/edges to simulate different algorithms.  
- 🔄 **Randomized Graph Generation** – Provides unique problem instances per student.  
- 📊 **Autogradable Submissions** – Automatically evaluates student responses.  
- 💡 **Usable in Multiple CS Courses** – Supports **introductory to advanced** graph algorithms.  

## Installation & Usage  

### **Requirements**  
- Python (>=3.8)  
- NetworkX, PyGraphviz, LXML, NumPy  
- PrairieLearn (for integration into an assessment platform)  
- Docker
  
### **Install Docker (if not already installed)**  
Follow the instructions from the official Docker documentation:  
- [Docker Installation Guide](https://docs.docker.com/get-docker/)  

Follow the official PrairieLearn local installation guide:  
- [PrairieLearn Local Setup](https://prairielearn.readthedocs.io/en/latest/installingLocal/)

---

## **Running the Project with Docker**  
To launch PrairieLearn and the interactive graph tool using Docker:

### **1. Clone this repository**  
```bash
git clone https://github.com/eldarhasanov079/pl-interactive-graph-v2/
```
### **2. Pull the PrairieLearn Docker image**  
```bash
docker pull --platform linux/x86_64 prairielearn/prairielearn
```
### **3. Run the PrairieLearn container**  
```bash
docker pull --platform linux/x86_64 prairielearn/prairielearn
```
### **3. Access PrairieLearn locally**  
Wait for <ins>http://localhost:3000</ins> to appear and click on it.

### **4. Load From Disk**  
Click "Load From Disk" in the top right corner and wait for it to run.

### **5. You are all set!**  
Try some of the example questions. Try changing their XML, or create your own questions (more information in the elements/pl-interactive-graph folder's README.md)


