import tensorflow as tf
import random

def euclidean_distance(a, b):
    """
    Computes the squared Euclidean distance between two tensors along the last axis.
    
    Args:
        a: Tensor of shape (..., embedding_dim)
        b: Tensor of shape (..., embedding_dim)
    
    Returns:
        Tensor of shape (...) representing the squared Euclidean distance.
    """
    return tf.reduce_sum(tf.square(a - b), axis=-1)


def prototypical_loss(support_embeddings, support_labels, query_embeddings, query_labels):
    """
    Computes the prototypical loss for few-shot learning.
    
    Args:
        support_embeddings: Tensor of shape (num_support, embedding_dim)
        support_labels: Tensor of shape (num_support,)
        query_embeddings: Tensor of shape (num_queries, embedding_dim)
        query_labels: Tensor of shape (num_queries,)
    
    Returns:
        Scalar tensor representing the loss value.
    """
    unique_classes = tf.unique(support_labels)[0]  # Extract unique class labels
    prototypes = []

    # Compute the class prototype by averaging embeddings of the same class
    for c in unique_classes:
        class_mask = tf.equal(support_labels, c)                                # Mask for selecting embeddings of class `c`
        class_embeddings = tf.boolean_mask(support_embeddings, class_mask)      # Select embeddings
        prototype = tf.reduce_mean(class_embeddings, axis=0)                    # Compute mean embedding
        prototypes.append(prototype)

    prototypes = tf.stack(prototypes)                                           # Stack class prototypes into a tensor (num_classes, embedding_dim)
    
    # Compute distances from query embeddings to each class prototype
    distances = euclidean_distance(tf.expand_dims(query_embeddings, 1), prototypes)  # Shape: (num_queries, num_classes)

    # Convert distances to class probabilities using softmax (negative distance for similarity)
    probabilities = tf.nn.softmax(-distances, axis=1)
    
    # Compute loss using sparse categorical cross-entropy
    loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(query_labels, probabilities))

    return loss


def sample_episode(dataset, num_classes=5, shots=5, queries=10):
    """
    Samples an episode for few-shot learning by selecting `num_classes` random classes
    and `shots` support + `queries` query examples per class.
    
    Args:
        dataset: TensorFlow dataset containing (images, labels)
        num_classes: Number of classes per episode
        shots: Number of support examples per class
        queries: Number of query examples per class
    
    Returns:
        Tuple of Tensors: (support_images, support_labels, query_images, query_labels)
    """
    class_samples = {}  # Dictionary to store images per class
    
    # Iterate through dataset to collect images per class
    for images, labels in dataset:
        for img, lbl in zip(images, labels.numpy()):  # Convert labels to numpy for indexing
            if lbl not in class_samples:
                class_samples[lbl] = []
            class_samples[lbl].append(img)

    # Randomly select `num_classes` classes for the episode
    sampled_classes = random.sample(class_samples.keys(), num_classes)
    
    support_images, support_labels = [], []
    query_images, query_labels = [], []

    # Sample support and query images for each selected class
    for class_idx, class_label in enumerate(sampled_classes):
        sampled_images = random.sample(class_samples[class_label], shots + queries)  # Select images
        support_images.extend(sampled_images[:shots])                                # First `shots` images for support set
        query_images.extend(sampled_images[shots:])                                  # Remaining `queries` images for query set
        support_labels.extend([class_idx] * shots)                                   # Assign new label to support set
        query_labels.extend([class_idx] * queries)                                   # Assign new label to query set

    return (
        tf.stack(support_images), tf.convert_to_tensor(support_labels),
        tf.stack(query_images), tf.convert_to_tensor(query_labels)
    )

