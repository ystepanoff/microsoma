package main

import (
	"fmt"
	"github.com/dominikbraun/graph"
	"github.com/dominikbraun/graph/draw"
	"github.com/google/uuid"
	"github.com/ystepanoff/microsoma/autograd"
	"github.com/ystepanoff/microsoma/neural"
	"os"
)

func main() {
	xs := [][]*autograd.Node{
		[]*autograd.Node{autograd.NewNode(2), autograd.NewNode(3), autograd.NewNode(-1)},
		[]*autograd.Node{autograd.NewNode(3), autograd.NewNode(-1), autograd.NewNode(0.5)},
		[]*autograd.Node{autograd.NewNode(0.5), autograd.NewNode(1.0), autograd.NewNode(1.0)},
		[]*autograd.Node{autograd.NewNode(1), autograd.NewNode(1), autograd.NewNode(-1)},
	}
	ys := []*autograd.Node{autograd.NewNode(1), autograd.NewNode(-1), autograd.NewNode(-1), autograd.NewNode(1)}
	yp := make([]*autograd.Node, 4)

	mlp := neural.NewMLP(3, []int{4, 4, 1})
	for i := range yp {
		yp[i] = mlp.Output(xs[i])[0]
	}
	fmt.Println(yp)
	loss := lossFunction(ys, yp)
	fmt.Println("loss", loss)

	mlp.Train(xs, ys, lossFunction, 10000, 0.0001)

	for i := range yp {
		yp[i] = mlp.Output(xs[i])[0]
	}
	fmt.Println(yp)
	newLoss := lossFunction(ys, yp)
	fmt.Println("new loss", newLoss)

	renderGraph(loss, "graph1.gv")
}

func lossFunction(ys []*autograd.Node, ypred []*autograd.Node) *autograd.Node {
	value := autograd.NewNode(0)
	for i := range ys {
		minusyp := ypred[i].Negate()
		diff := ys[i].Add(minusyp)
		diffSq := diff.Mul(diff)
		value = value.Add(diffSq)
	}
	return value
}

func renderGraph(root *autograd.Node, fileName string) {
	g := graph.New(graph.StringHash, graph.Directed())
	buildGraph(root, &g)
	file, _ := os.Create(fileName)
	_ = draw.DOT(g, file)
}

func buildGraph(root *autograd.Node, g *graph.Graph[string, string]) {
	if err := (*g).AddVertex(root.Id, graph.VertexAttribute("label", root.String())); err != nil {
		return
	}
	if root.Operation != "" {
		v := uuid.NewString()
		_ = (*g).AddVertex(v, graph.VertexAttribute("label", root.Operation))
		_ = (*g).AddEdge(root.Id, v)
		for _, child := range root.Children {
			buildGraph(child, g)
			_ = (*g).AddEdge(v, child.Id)
		}
	}
}
