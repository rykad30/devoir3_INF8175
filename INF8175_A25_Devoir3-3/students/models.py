import nn
from backend import PerceptronDataset, RegressionDataset, DigitClassificationDataset


class PerceptronModel(object):
    def __init__(self, dimensions: int) -> None:
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self) -> nn.Parameter:
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x: nn.Constant) -> nn.Node:
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        return nn.DotProduct(x, self.w)

    def get_prediction(self, x: nn.Constant) -> int:
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        score = nn.as_scalar(self.run(x))
        if score >= 0:
            return 1
        else:
            return -1

    def train(self, dataset: PerceptronDataset) -> None:
        """
        Train the perceptron until convergence.

        """
        inOrder = False
        while not inOrder:
            inOrder = True # assume all correct until we find one mistake
            for x, y in dataset.iterate_once(1) :
                y_true = nn.as_scalar(y)
                if self.get_prediction(x) != y_true:
                    inOrder = False # found a mistake
                    self.w.update(x, y_true)

        return



class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self) -> None:
        # Initialize your model parameters here
        hidden_size = 128  # Increased size for better approximation of sin(x)
        
        # Input to hidden layer: (1 input feature -> hidden_size neurons)
        self.W1 = nn.Parameter(1, hidden_size)
        self.b1 = nn.Parameter(1, hidden_size)
        
        # Hidden to output layer: (hidden_size neurons -> 1 output feature)
        self.W2 = nn.Parameter(hidden_size, 1)
        self.b2 = nn.Parameter(1, 1)

    def run(self, x: nn.Constant) -> nn.Node:
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        # First layer: Linear transformation + bias
        hidden_linear = nn.Linear(x, self.W1)
        hidden_with_bias = nn.AddBias(hidden_linear, self.b1)
        
        # Activation function: ReLU
        hidden_activated = nn.ReLU(hidden_with_bias)
        
        # Output layer: Linear transformation + bias
        output_linear = nn.Linear(hidden_activated, self.W2)
        output = nn.AddBias(output_linear, self.b2)
        
        return output

    def get_loss(self, x: nn.Constant, y: nn.Constant) -> nn.Node:
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        predictions = self.run(x)
        return nn.SquareLoss(predictions, y)

    def train(self, dataset: RegressionDataset) -> None:
        """
        Trains the model.
        """
        learning_rate = 0.01
        batch_size = 20  # On assume que tous les datasets ont des tailles multiples de 20
        
        num_epochs = 2000
        target_loss = 0.02 # Stop when loss reaches target
        
        for epoch in range(num_epochs):
            total_loss = 0
            batch_count = 0
            
            for x_batch, y_batch in dataset.iterate_once(batch_size):
                # Compute loss for this batch
                loss = self.get_loss(x_batch, y_batch)
                
                # Compute gradients with respect to all parameters
                parameters = [self.W1, self.b1, self.W2, self.b2]
                grads = nn.gradients(loss, parameters)
                
                # Update parameters using gradient descent
                for param, grad in zip(parameters, grads):
                    param.update(grad, -learning_rate)
                
                total_loss += nn.as_scalar(loss)
                batch_count += 1
            
            # Calculate average loss
            if batch_count > 0:
                avg_loss = total_loss / batch_count
                
                # Print progress monitoring
                if epoch % 200 == 0:
                    print(f"Epoch {epoch}, Average Loss: {avg_loss:.6f}")
                
                # Early stopping if target loss is achieved
                if avg_loss <= target_loss:
                    print(f"Target loss reached! Epoch {epoch}, Loss: {avg_loss:.6f}")
                    break
        
        # Final loss evaluation
        final_loss = nn.as_scalar(self.get_loss(
            nn.Constant(dataset.x), 
            nn.Constant(dataset.y)
        ))
        print(f"Training completed. Final loss: {final_loss:.6f}")


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self) -> None:
        # Initialize your model parameters here
        hidden_size1 = 128
        hidden_size2 = 64
        
        # Couche 1: 784 entrées -> 128 neurones
        self.W1 = nn.Parameter(784, hidden_size1)
        self.b1 = nn.Parameter(1, hidden_size1)
        
        # Couche 2: 128 -> 64 neurones
        self.W2 = nn.Parameter(hidden_size1, hidden_size2)
        self.b2 = nn.Parameter(1, hidden_size2)
        
        # Couche de sortie: 64 -> 10 classes
        self.W3 = nn.Parameter(hidden_size2, 10)
        self.b3 = nn.Parameter(1, 10)

    def run(self, x: nn.Constant) -> nn.Node:
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        # Première couche: Linear + Bias + ReLU
        hidden1_linear = nn.Linear(x, self.W1)
        hidden1_with_bias = nn.AddBias(hidden1_linear, self.b1)
        hidden1_activated = nn.ReLU(hidden1_with_bias)
        
        # Deuxième couche: Linear + Bias + ReLU
        hidden2_linear = nn.Linear(hidden1_activated, self.W2)
        hidden2_with_bias = nn.AddBias(hidden2_linear, self.b2)
        hidden2_activated = nn.ReLU(hidden2_with_bias)
        
        # Couche de sortie: Linear + Bias (pas d'activation pour les logits)
        output_linear = nn.Linear(hidden2_activated, self.W3)
        output = nn.AddBias(output_linear, self.b3)
        
        return output

    def get_loss(self, x: nn.Constant, y: nn.Constant) -> nn.Node:
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        logits = self.run(x)
        return nn.SoftmaxLoss(logits, y)

    def train(self, dataset: DigitClassificationDataset) -> None:
        """
        Trains the model.
        """
        learning_rate = 0.1
        batch_size = 20
        num_epochs = 5 # Pas trop haut sinon trop long
        best_accuracy = 0
        patience = 3  # Arrêt anticipé si pas d'amélioration après 3 époques
        patience_counter = 0
        
        for epoch in range(num_epochs):
            total_loss = 0
            batch_count = 0
            
            # Phase d'entraînement
            for x_batch, y_batch in dataset.iterate_once(batch_size):
                # Compute loss for this batch
                loss = self.get_loss(x_batch, y_batch)
                
                # Compute gradients with respect to all parameters
                parameters = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
                grads = nn.gradients(loss, parameters)
                
                # Update parameters using gradient descent
                for param, grad in zip(parameters, grads):
                    param.update(grad, -learning_rate)
                
                total_loss += nn.as_scalar(loss)
                batch_count += 1
            
            # Calcul de la précision sur l'ensemble de validation
            validation_accuracy = dataset.get_validation_accuracy()
            
            # Affichage des métriques
            avg_loss = total_loss / batch_count if batch_count > 0 else total_loss
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}, Validation Accuracy: {validation_accuracy:.3f}")
            
            # Arrêt anticipé basé sur la précision de validation
            if validation_accuracy > best_accuracy:
                best_accuracy = validation_accuracy
                patience_counter = 0
                print(f"  → New best accuracy: {best_accuracy:.3f}")
            else:
                patience_counter += 1
                print(f"  → No improvement for {patience_counter} epoch(s)")
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
            # Réduction du learning rate si stagnation
            if patience_counter >= 2:
                learning_rate *= 0.8
                print(f"  → Reducing learning rate to {learning_rate:.4f}")
        
        # Évaluation finale
        final_accuracy = dataset.get_validation_accuracy()
        print(f"Training completed. Final validation accuracy: {final_accuracy:.3f}")
