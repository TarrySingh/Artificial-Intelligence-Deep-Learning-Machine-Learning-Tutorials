/** Simple Neural Network library that can only create neural networks of exactly 3 layers */
class NeuralNetwork {

    /**
     * Takes in the number of input nodes, hidden node and output nodes
     * @constructor
     * @param {number} input_nodes
     * @param {number} hidden_nodes 
     * @param {number} output_nodes 
     */
    constructor(input_nodes, hidden_nodes, output_nodes) {

        this.input_nodes = input_nodes;
        this.hidden_nodes = hidden_nodes;
        this.output_nodes = output_nodes;

        // Initialize random weights
        this.input_weights = tf.randomNormal([this.input_nodes, this.hidden_nodes]);
        this.output_weights = tf.randomNormal([this.hidden_nodes, this.output_nodes]);
    }

    /**
     * Takes in a 1D array and feed forwards through the network
     * @param {array} - Array of inputs
     */

    predict(user_input) {
        let output;
        tf.tidy(() => {
            /* Takes a 1D array */
            let input_layer = tf.tensor(user_input, [1, this.input_nodes]);
            let hidden_layer = input_layer.matMul(this.input_weights).sigmoid();
            let output_layer = hidden_layer.matMul(this.output_weights).sigmoid();
            output = output_layer.dataSync();
        });
        return output;
    }

    /**
     * Returns a new network with the same weights as this Neural Network
     * @returns {NeuralNetwork}
     */
    clone() {
        let clonie = new NeuralNetwork(this.input_nodes, this.hidden_nodes, this.output_nodes);
        clonie.dispose();
        clonie.input_weights = tf.clone(this.input_weights);
        clonie.output_weights = tf.clone(this.output_weights);
        return clonie;
    }

    /**
     * Dispose the input and output weights from the memory
     */
    dispose() {
        this.input_weights.dispose();
        this.output_weights.dispose();
    }
}