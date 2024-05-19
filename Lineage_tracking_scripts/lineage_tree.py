# @title
import pygraphviz as pgv
from PIL import Image
import io
import numpy as np
data = global_id[0:int(np.max(global_id[:,0]))+1,0:2]
# Load your data from a NumPy array
# Replace the following line with your actual data
#data = np.array([[1, -1], [2, 1], [3, 2], [4, 1], [5, 2]])

# Create a directed graph
def create_tree(color_map, data, timing_id, saving_address, frame_increment=1, time_unit="day"):
  G = pgv.AGraph(strict=False, directed=True)

  # Create a set to keep track of nodes with no ancestors
  ancestors = set()

  # Iterate through your data and add all nodes to the graph
  for i, row in enumerate(data):
      individual_id = str(int(row[0]))  # Convert "id" to a string
      G.add_node(individual_id, label=str(individual_id))

      # Check if the individual has no ancestor (inheritance = -1)
      if row[1] == -1:
          ancestors.add(individual_id)
      color = "#{:02X}{:02X}{:02X}".format(
          int(color_map[i,0] * 255), int(color_map[i,1] * 255), int(color_map[i,2] * 255)
      )
      G.get_node(individual_id).attr['color'] = color
      G.get_node(individual_id).attr['style'] = 'filled'
      G.get_node(individual_id).attr['fontcolor'] = 'white'
      G.get_node(individual_id).attr['fontstyle'] = 'bold'

  # Iterate through your data to add edges for those with valid inheritance values
  for jj, row in enumerate(data):
      individual_id = str(int(row[0]))  # Convert "id" to a string
      inheritance = str(int(row[1]))  # Convert "inheritance" to a string

      # Check if the individual has an ancestor
      if inheritance != '-1':
          if timing_id[jj]>1:
            edge_label = f" t = {int(timing_id[jj])*frame_increment} {time_unit}s"
          else:
            edge_label = f" t = {int(timing_id[jj])*frame_increment} {time_unit}"
          G.add_edge(inheritance, individual_id, label=edge_label)


  # Find the root nodes (individuals with no parents)
  root_nodes = [node for node in ancestors]

  # Layout the graph
  G.layout(prog='dot')

  # Save the graph to a file
  graph_file = saving_address+'/'+f'lineage_tree.png'
  G.draw(graph_file)

  # Display the graph
  #with Image.open(graph_file) as img:
  #    img.show()
 #vv=plt.imread()
  #plt.imshow(vv)
  #plt.axis('off');
  #plt.show()


