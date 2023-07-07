package autograd

import (
	"fmt"
	"github.com/google/uuid"
	"math"
)

type Node struct {
	Value     float64
	Grad      float64
	Children  []*Node
	Operation string
	Id        string
}

func NewNode(value float64) *Node {
	return &Node{
		Value: value,
		Id:    uuid.NewString(),
	}
}

func (a *Node) Add(b *Node) *Node {
	return &Node{
		Value:     a.Value + b.Value,
		Children:  []*Node{a, b},
		Operation: "+",
		Id:        uuid.NewString(),
	}
}

func (a *Node) Mul(b *Node) *Node {
	return &Node{
		Value:     a.Value * b.Value,
		Children:  []*Node{a, b},
		Operation: "*",
		Id:        uuid.NewString(),
	}
}

func (a *Node) Tanh() *Node {
	return &Node{
		Value:     math.Tanh(a.Value),
		Children:  []*Node{a},
		Operation: "tanh",
		Id:        uuid.NewString(),
	}
}

func (a *Node) Negate() *Node {
	return &Node{
		Value:     -a.Value,
		Children:  []*Node{a},
		Operation: "neg",
		Id:        uuid.NewString(),
	}
}

func (node *Node) String() string {
	return fmt.Sprintf("[Value: %.2f | Grad: %.2f]", node.Value, node.Grad)
}
