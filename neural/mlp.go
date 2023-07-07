package neural

import "github.com/ystepanoff/microsoma/autograd"

type MLP struct {
	NInputs  int
	NOutputs []int
	Layers   []*Layer
}

func NewMLP(nInputs int, nOutputs []int) *MLP {
	sizes := append([]int{nInputs}, nOutputs...)
	layers := make([]*Layer, len(nOutputs))
	for i := 0; i < len(nOutputs); i++ {
		layers[i] = NewLayer(sizes[i], sizes[i+1])
	}
	return &MLP{
		NInputs:  nInputs,
		NOutputs: nOutputs,
		Layers:   layers,
	}
}

func (mlp *MLP) Output(x []*autograd.Node) []*autograd.Node {
	for _, layer := range mlp.Layers {
		x = layer.Output(x)
	}
	return x
}

func (mlp *MLP) Parameters() []*autograd.Node {
	params := make([]*autograd.Node, 0)
	for _, layer := range mlp.Layers {
		params = append(params, layer.Parameters()...)
	}
	return params
}

func (mlp *MLP) Train(
	xs [][]*autograd.Node,
	ys []*autograd.Node,
	lossFunction func([]*autograd.Node, []*autograd.Node) *autograd.Node,
	args ...interface{},
) {
	nsamples := len(xs)
	steps := 1000
	h := 0.01
	if len(args) > 0 {
		steps = args[0].(int)
	}
	if len(args) > 1 {
		h = args[1].(float64)
	}
	for step := 0; step < steps; step++ {
		ypred := make([]*autograd.Node, nsamples)
		for i := range ypred {
			ypred[i] = mlp.Output(xs[i])[0]
		}
		loss := lossFunction(ys, ypred)
		loss.Propagate()
		for _, p := range mlp.Parameters() {
			p.Value -= h * p.Grad
		}
	}
}
