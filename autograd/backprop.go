package autograd

import "math"

func (root *Node) Propagate() {
	zeroGrad(root)
	root.Grad = 1
	propagate(root)
}

func propagate(root *Node) {
	switch root.Operation {
	case "+":
		for _, child := range root.Children {
			child.Grad += 1 * root.Grad
		}
	case "*":
		for i, child := range root.Children {
			prod := 1.0
			for j, c := range root.Children {
				if i != j {
					prod *= c.Value
				}
			}
			child.Grad += prod * root.Grad
		}
	case "tanh":
		root.Children[0].Grad = (1.0 - math.Pow(root.Value, 2)) * root.Grad
	case "neg":
		root.Children[0].Grad = -root.Grad
	}
	for _, child := range root.Children {
		propagate(child)
	}
}

func zeroGrad(root *Node) {
	root.Grad = 0
	for _, child := range root.Children {
		zeroGrad(child)
	}
}
