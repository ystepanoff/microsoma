package neural

import (
	"github.com/ystepanoff/microsoma/autograd"
	"math/rand"
	"time"
)

type Neuron struct {
	NInputs int
	W       []*autograd.Node
	B       *autograd.Node
}

func NewNeuron(nInputs int) *Neuron {
	rand.Seed(time.Now().UnixNano())
	w := make([]*autograd.Node, nInputs)
	for i := range w {
		w[i] = autograd.NewNode(-1 + rand.Float64()*2)
	}
	return &Neuron{
		NInputs: nInputs,
		W:       w,
		B:       autograd.NewNode(-1 + rand.Float64()*2),
	}
}

func (neuron *Neuron) Output(x []*autograd.Node) *autograd.Node {
	out := neuron.B
	for i, w := range neuron.W {
		out = out.Add(w.Mul(x[i]))
	}
	return out.Tanh()
}

func (neuron *Neuron) Parameters() []*autograd.Node {
	return append(neuron.W, neuron.B)
}

type Layer struct {
	NInputs  int
	NOutputs int
	Neurons  []*Neuron
}

func NewLayer(nInputs int, nOutputs int) *Layer {
	neurons := make([]*Neuron, nOutputs)
	for i := range neurons {
		neurons[i] = NewNeuron(nInputs)
	}
	return &Layer{
		NInputs:  nInputs,
		NOutputs: nOutputs,
		Neurons:  neurons,
	}
}

func (layer *Layer) Output(x []*autograd.Node) []*autograd.Node {
	outs := make([]*autograd.Node, len(layer.Neurons))
	for i, neuron := range layer.Neurons {
		outs[i] = neuron.Output(x)
	}
	return outs
}

func (layer *Layer) Parameters() []*autograd.Node {
	params := make([]*autograd.Node, 0)
	for _, neuron := range layer.Neurons {
		params = append(params, neuron.Parameters()...)
	}
	return params
}
