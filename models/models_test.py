from math import ceil

import torch
from experiments.utils.training import build_TransformerCA
from models.PortfolioCrossFormer import build_CrossFormer
from models.PortfolioMASTER import build_MASTER

MODEL_NAME = "MASTER"  # TransformerCA/CrossFormer/MASTER
#
if __name__ == '__main__':
    # Transformer Cross Attention
    if MODEL_NAME == 'TransformerCA':
        batch_size = 64
        lookback = 20
        stock_amount = 10
        features = 5
        example_d_model = 64
        example_n_heads = 4
        example_d_ff = 256
        example_dropout = 0.1
        example_num_layers = 2
        example_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = build_TransformerCA(
            stock_amount=stock_amount,
            financial_features_amount=features,
            lookback=lookback,
            d_model=example_d_model,
            n_heads=example_n_heads,
            d_ff=example_d_ff,
            dropout=example_dropout,
            num_encoder_layers=example_num_layers,
            device=example_device
        )

        dummy_input = torch.randn(batch_size, lookback, stock_amount, features, device=example_device)
        print(f"\nTesting model with input shape: {dummy_input.shape}")

        try:
            model.eval()
            with torch.no_grad():
                output_returns = model(dummy_input)
            print(f"Model output shape: {output_returns.shape}")
            print("Model forward pass successful!")
        except Exception as e:
            print(f"Error during model forward pass: {e}")
            import traceback

            traceback.print_exc()

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params:,}")

    # Crossformer
    if MODEL_NAME == 'CrossFormer':
        # Parametry
        batch_size = 32
        lookback = 96
        stock_amount = 10
        financial_features = 5
        seg_len = 24
        win_size = 2
        factor = 10
        d_model = 128
        d_ff = 256
        n_heads = 4
        e_layers = 2
        dropout = 0.1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        if lookback % seg_len != 0:
            print(
                f"Warning: lookback ({lookback}) is not perfectly divisible by seg_len ({seg_len}). Padding will be applied.")

        input_data = torch.randn(batch_size, lookback, stock_amount, financial_features).to(device)

        model = build_CrossFormer(
            stock_amount=stock_amount,
            financial_features=financial_features,
            in_len=lookback,
            seg_len=seg_len,
            win_size=win_size,
            factor=factor,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            e_layers=e_layers,
            dropout=dropout,
            device=device
        ).to(device)

        # Sprawdzenie poprawności obliczenia final_seg_num w modelu
        calculated_final_seg = ceil(lookback / seg_len)
        for _ in range(e_layers - 1):
            if win_size > 1:
                calculated_final_seg = ceil(calculated_final_seg / win_size)
        print(f"Calculated final_seg_num outside model: {calculated_final_seg}")
        print(f"Final_seg_num inside model: {model.final_seg_num}")
        assert model.final_seg_num == calculated_final_seg, "Mismatch in final_seg_num calculation!"

        output_weights = model(input_data)

        print(f"Input shape: {input_data.shape}")
        print(f"Output shape: {output_weights.shape}")
        print(f"Output weights (first batch sample): {output_weights[0]}")
        print(f"Sum of weights (first batch sample): {output_weights[0].sum()}")

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params:,}")

    if MODEL_NAME == 'MASTER':
        # Example input data characteristics
        batch_size = 64
        lookback = 20       # T
        stock_amount = 10   # N
        features = 5        # F

        # Example parameters
        example_d_model = 64
        example_t_heads = 4
        example_s_heads = 4
        example_t_dropout = 0.1
        example_s_dropout = 0.1
        example_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build the model
        model = build_MASTER(
            stock_amount=stock_amount,
            financial_features_amount=features,
            lookback=lookback,
            d_model=example_d_model,
            t_n_heads=example_t_heads,
            s_n_heads=example_s_heads,
            t_dropout=example_t_dropout,
            s_dropout=example_s_dropout,
            device=example_device
        )

        # Test with dummy data
        dummy_input = torch.randn(batch_size, lookback, stock_amount, features, device=example_device)
        print(f"\nTesting model with input shape: {dummy_input.shape}") # [B, T, N, F]

        # Pass data through the model
        try:
            model.eval() # Set to evaluation mode
            with torch.no_grad():
                 output_returns = model(dummy_input)
            print(f"Model output shape: {output_returns.shape}") # Expected: [B, N, 1]
            print("Model forward pass successful!")
        except Exception as e:
            print(f"Error during model forward pass: {e}")
            import traceback
            traceback.print_exc()

        # Check number of parameters
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params:,}")
