
# pl-interactive-graph for PrairieLearn

## Overview
`pl-interactive-graph` is a custom interactive element for PrairieLearn, designed for creating and interacting with graph-based questions. It supports various functionalities like graph traversal, visualization, and interactive node/edge selection. The element is built on top of the existing `pl-graph` element.

## Usage
To use the `pl-interactive-graph` element in your PrairieLearn course:

1. **Include the Element in Your Question**: Embed the custom element tag `<pl-interactive-graph>`in your question HTML file.
2. **Define the Graph**: Specify the graph structure within the tag using your desired graph generation method. Use DOT language to specify your graph, to learn how to use DOT language, navigate to https://graphviz.org/. *Note:* If you would like randomized graphs, you do not need to use the DOT language, feel free to leave the body blank.
3.  **Set Attributes**: Customize the behavior and appearance of the graph using XML attributes. Attributes numbered 9-16 are used for random graphs. If you are not using a random graph, these attributes add nothing. Further, attributes 16 and on are part of the existing `pl-graph` element. The element supports a variety of attributes to cater to different question types and requirements:

	1. `preserve-ordering`: String. If set to `"True"`, it requires the answer sequence to match exactly.

	2. `answers`: String. String of an array of node labels representing the correct answer. (Example: '["A","B","C"]')

	3. `partial-credit`: String. If set to `"True"`, it allows partial credit for partially correct sequences.

	4. `node-fill-color`: String. If set, changes the fill color of selected nodes to that color (default is "red").

	5. `edge-fill-color`: String. If set, changes the fill color of selected edges to that color (default is "green").
	
 	6. `select-nodes`: Boolean. If set to `True`, it allows the user to click on nodes for selection for interaction.

	7. `select-edges`: Boolean. If set to `True`, it allows the user to click on edges for selection for interaction. 

	8. `random-graph`: Boolean. If set to `True`, random graphs will be generated.

	9. `directed-random`: Boolean. If set to `True`, random graphs will be generated with directed edges.

	10. `min-nodes`: Integer. Defines the minimum number of nodes in a random graph.

	11. `max-nodes`: Integer. Defines the maximum number of nodes in a random graph.

	12. `min-edges`: Integer. Defines the minimum number of edges in a random graph.

	13. `max-edges`: Integer. Defines the maximum number of edges in a random graph.

	14. `weighted`: Boolean. Specifies if a random graph should have random edge weights.

	15. `tree`: Boolean. Specifies if the random graph should be a tree.

    16. `connected`: Boolean. Specifies if the random graph should be connected.

	17.  `directed`: Boolean. Whether to treat edges in an adjacency matrix as directed or undirected. If set to false, then edges will be rendered as undirected. The input adjacency matrix must be symmetric if this is set to false.

	18.  `engine`: String. The rendering engine to use; supports circo, dot, fdp, neato, osage, and twopi.

	19.  `params-name`: String. The the name of a parameter containing the data to use as input. Data type to use depends on params-type attribute.

	20.  `params-name-labels`: String. When using an adjacency matrix, the parameter that contains the labels for each node.

   	21.  `params-type`: String. Type of graph representation, e.g., `"adjacency-matrix"` or `"networkx"`.

	22.  `weights`: Boolean. When using an adjacency matrix, whether or not to show the edge weights. By default will automatically show weights for stochastic matrices (when they are not binary 0/1).

	23.  `weights-digits`: Integer. When using an adjacency matrix, how many digits to show for the weights.
  
	24.  `negative-weights`: Boolean. Whether to recognize negative weights in an adjacency matrix. If set to false, then all weights at most 0 are ignored (not counted as an edge). If set to true, then all weights that are not None are recognized.

	25.  `weights-presentation-type`: String. Number display format for the weights when using an adjacency matrix. If presentation-type is 'sigfig', each number is formatted using the to_precision module to digits significant figures. Otherwise, each number is formatted as {:.{digits}{presentation-type}}.

	26.  `log-warnings`: Boolean. Whether to log warnings that occur during Graphviz rendering.

Some of the attributes have been inherited from pl-graph, here is more information on those specific inherited attributes: https://prairielearn.readthedocs.io/en/latest/elements/#pl-graph-element

4. **Modify server.py if Needed**: Determine how would you want to grade the question. To access the order given by the student as the nodes were clicked, you can do student_answer = data["submitted_answers"]["selectedNodes"]. Note: If you have used custom attributes, like preserve-ordering or answers, this part might be different. There is existing autograding if answers are provided in the `<pl-interactive-graph>` as `<pl-interactive-graph answers='["A","B","C"]'>` for edges, `<pl-interactive-graph answers='["A--B","B--C"]'>` for undirected edges and `<pl-interactive-graph answers='["A->B","B->C"]'>` for directed edges as per DOT language. 


## Description
The students will be presented with a graph of your specified structure and each node and edge will be clickable according to your selection. Students can click the nodes/edges and depending on the element attribute values, the order might matter (they can also unclick nodes). A list of clicked nodes and/or edges in the corresponding order will be shown right under the graph. When the students click "submit" the element will record the clicked nodes and provide them to the backend. The backend has 2 options for grading, one with provision of direct answers through the 'answers' attribute and automatic through the 'grading' attribute, which allows to select of an algorithm to automatically grade the submission for both random and provided graph. 

## Suggested Use
This element is not only limited to purely graph traversal questions. Some of the possible problems that could be modelled by this element are (but not limited to): Network Flow, Finite State Machines, Markov Chains, Bayes Nets, Pathfinding Algorithms, Routing Protocols etc. 

## Example
Different examples have been included in the questions folder.

Here's an example of how you might use `pl-interactive-graph` in a question about graph traversal:

```html
<p>You are provided a graph below. Click on the nodes in the order they will be selected if we run Breadth-First Search (BFS) algorithm on this graph. You can see the order of your clicked list in a list under the graph and you are allowed to unclick and deselect nodes at any point. Press "Save & Grade" to submit your answer.</p>
<pl-question-panel>  
  <pl-interactive-graph node_fill_color="red" preserve-ordering="True" answers='["A", "B", "C", "D"]' partial-credit="True" select_nodes="True" select_edges="False">
    digraph G {
      A -> B;
      A -> C;
      B -> D;
      B -> E;
      C -> F;
      C -> G;
      E -> H;
  }
  
  
  </pl-interactive-graph>
</pl-question-panel>

