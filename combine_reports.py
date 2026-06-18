import os

out_path = "combined_hierarchical_topology_report.md"

rfdetr_path = "topology_hierarchical_output/rfdetr/hierarchical_topology_report.md"
dinov2_path = "topology_hierarchical_output/dinov2/hierarchical_topology_report.md"

with open(out_path, "w") as f:
    f.write("# Combined Hierarchical Clustering Topology Report (RF-DETR vs DINOv2)\n\n")
    
    f.write("This report constructs a tree based on standard Agglomerative Hierarchical Clustering. Because linkage requires symmetric distances, we use the average coverage between the two directions. Each internal node shows the dataset that provides the best coverage acting as the parent.\n\n")
    
    f.write("---\n\n")
    f.write("# Part 1: RF-DETR Hierarchical Topologies\n\n")
    
    with open(rfdetr_path, "r") as rf:
        # Skip the first few lines of header to avoid duplicate text
        lines = rf.readlines()[6:]
        f.writelines(lines)
        
    f.write("\n---\n\n")
    f.write("# Part 2: DINOv2 Hierarchical Topologies\n\n")
    
    with open(dinov2_path, "r") as df:
        lines = df.readlines()[6:]
        f.writelines(lines)

print("Combined report generated!")
