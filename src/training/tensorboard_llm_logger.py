import torch
from torch.utils.tensorboard import SummaryWriter
import math
import time
import numpy as np
import subprocess
import os

from src.models.model_upgrades import postprocess_french_text

class LLMTensorboardLogger:
    def __init__(self, log_dir="runs/llm_logs", port=6006, launch_tb=True):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.start_time = time.time()
        self.total_tokens = 0
        self.tb_process = None

        # Lance TensorBoard si demandé
        if launch_tb:
            self.launch_tensorboard(logdir=log_dir, port=port)


    # ---------------------------------------------------
    # Lancement automatique de TensorBoard
    # ---------------------------------------------------
    def launch_tensorboard(self, logdir="runs", port=6006):

        # Ferme les anciens processus
        subprocess.call(["pkill", "-f", "tensorboard"])

        print("Lancement automatique de TensorBoard...")

        # Fermer une ancienne instance s'il y en a une
        if self.tb_process is not None:
            try:
                self.tb_process.terminate()
            except Exception:
                pass

        # Lancer TensorBoard en background
        self.tb_process = subprocess.Popen(
            ["tensorboard", f"--logdir={logdir}", f"--port={port}", "--bind_all"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Laisser le serveur démarrer
        time.sleep(2)

        print(f"TensorBoard lancé : http://localhost:{port}")


    # ---------------------------------------------------
    # fermeture de TensorBoard 
    # ---------------------------------------------------
    def close(self):
        """Ferme proprement TensorBoard et le writer."""
        print("Fermeture de TensorBoard...")

        # Fermer le SummaryWriter
        try:
            self.writer.flush()
            self.writer.close()
        except Exception:
            pass

        # Fermer TensorBoard
        if self.tb_process is not None:
            try:
                self.tb_process.terminate()
                self.tb_process.wait(timeout=3)
                print("TensorBoard fermé proprement.")
            except Exception:
                print("Échec terminate() → tentative kill()")
                try:
                    self.tb_process.kill()
                    print("TensorBoard tué.")
                except Exception:
                    print("Impossible de fermer TensorBoard.")
    
    # ---------------------------------------------------
    # SCALARS
    # ---------------------------------------------------
    def log_train_loss(self, loss, step):
        self.writer.add_scalar("train/loss", loss, step)
        self.writer.add_scalar("train/perplexity", math.exp(loss), step)

    def log_eval_loss(self, loss, step):
        self.writer.add_scalar("eval/loss", loss, step)
        self.writer.add_scalar("eval/perplexity", math.exp(loss), step)

    def log_lr(self, optimizer, step):
        lr = optimizer.param_groups[0]["lr"]
        self.writer.add_scalar("train/learning_rate", lr, step)

    # ---------------------------------------------------
    # NORMS
    # ---------------------------------------------------
    def log_grad_norm(self, model, step):
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.writer.add_scalar("train/grad_norm", total_norm, step)

    def log_weight_norm(self, model, step):
        total_norm = 0.0
        for p in model.parameters():
            param_norm = p.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.writer.add_scalar("train/weight_norm", total_norm, step)

    # ---------------------------------------------------
    # TEXT SAMPLES
    # ---------------------------------------------------
    def log_generated_text(self, tokenizer, model, step, max_new_tokens=50):
        try:
            model.eval()

            prompt = "Le planning pour aujourd'hui est :"
            enc = tokenizer(prompt, return_tensors="pt")

            # récupérer uniquement input_ids
            input_ids = enc["input_ids"].to(next(model.parameters()).device)

            with torch.inference_mode():
                # MINI-GPT N'ACCEPTE PAS attention_mask, token_type_ids, etc.
                out = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=1.0,   # tu peux en mettre un autre
                    top_k=50           # tu peux désactiver en mettant None
                )

            text = tokenizer.decode(out[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            text = postprocess_french_text(text)
            self.writer.add_text("samples/generated", text, step)

        except Exception as e:
            print(f"[TensorBoard Logger] Impossible de générer un texte : {e}")

        finally:
            model.train()



    # ---------------------------------------------------
    # HISTOGRAMS
    # ---------------------------------------------------
    def log_histograms(self, model, step):
        for name, param in model.named_parameters():
            self.writer.add_histogram(f"weights/{name}", param.data.cpu().numpy(), step)

    # ---------------------------------------------------
    # SPEED MONITOR
    # ---------------------------------------------------
    def log_tokens_per_second(self, tokens_this_step, step):
        self.total_tokens += tokens_this_step
        elapsed = time.time() - self.start_time
        tps = self.total_tokens / max(1e-8, elapsed)
        self.writer.add_scalar("train/tokens_per_second", tps, step)
