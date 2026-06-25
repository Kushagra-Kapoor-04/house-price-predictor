import numpy as np

def train_model(model, X_train, y_train, epochs, learning_rate, X_val=None, y_val=None, verbose=True):
    """
    Executes the training loop over specified number of epochs.
    Logs and records training (and optionally validation) loss history.
    """
    loss_history = []
    val_loss_history = []
    
    for epoch in range(1, epochs + 1):
        # 1. Forward pass
        y_pred = model.forward(X_train)
        
        # 2. Compute training loss
        loss = model.compute_loss(y_pred, y_train)
        loss_history.append(loss)
        
        # 3. Compute gradients
        dw, db = model.compute_gradients(X_train, y_pred, y_train)
        
        # 4. Update parameters
        model.update_parameters(dw, db, learning_rate)
        
        # 5. Handle validation loss if validation data is provided
        if X_val is not None and y_val is not None:
            y_val_pred = model.forward(X_val)
            val_loss = model.compute_loss(y_val_pred, y_val)
            val_loss_history.append(val_loss)
        
        # Verbose logging
        if verbose and (epoch == 1 or epoch % (max(1, epochs // 10)) == 0 or epoch == epochs):
            log_str = f"Epoch {epoch}/{epochs} - Loss: {loss:.6f}"
            if X_val is not None and y_val is not None:
                log_str += f" - Val Loss: {val_loss:.6f}"
            print(log_str)
            
    return loss_history, val_loss_history
