import matplotlib.pyplot as plt
import kruskal
import prim
import time

num_of_iterations = 10
completeness = 1
nums_of_vertexes = [20, 50, 100, 200, 250, 500, 700]
# nums_of_vertexes = [10, 15, 20, 50]
min_prim = []
min_kruskal = []

for amount in nums_of_vertexes:
    time_holder = []
    for i in range(num_of_iterations):
        start = time.time()
        prim.step(prim.gnp_random_connected_graph(amount, completeness))
        end = time.time()
        time_holder.append(end-start)
    min_time_taken = min(time_holder)
    min_prim.append(round(min_time_taken, 4))

for amount in nums_of_vertexes:
    time_holder = []
    for i in range(num_of_iterations):
        start = time.time()
        kruskal.kruskal(kruskal.gnp_random_connected_graph(amount, completeness))
        end = time.time()
        time_holder.append(end-start)
    min_time_taken = min(time_holder)
    min_kruskal.append(round(min_time_taken, 4))

y_prim = min_prim
y_kruskal = min_kruskal

plt.style.use("dark_background")
fig, ax = plt.subplots()

ax.plot(nums_of_vertexes, y_prim, marker = "o", label = "Prim")
ax.plot(nums_of_vertexes, y_kruskal, marker = "o", label = "Kruskal")
ax.set_title("Comparison", fontsize = 20)
ax.set_xlabel("Number of nodes")
ax.set_ylabel("Execution time, secs")
ax.legend(loc = "upper left")
ax.grid(linestyle = "--", linewidth = 0.5, color = "grey")
plt.show()
