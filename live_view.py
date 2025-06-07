diff --git a/live_view.py b/live_view.py
index cf1ac3a070f1e45c2773be007ce5301cd0dd871d..18c3adbe807264d6692f46251c50331c79032ba8 100644
--- a/live_view.py
+++ b/live_view.py
@@ -60,86 +60,80 @@ def md5(path: Path) -> str:
 pygame.init()
 FONT = pygame.font.SysFont("consolas", 28, bold=True)
 
 env = TetrisEnv()
 env.render("human")   # opens Pygame window
 env.reset()           # first reset
 
 def overlay(text: str):
     """
     Draw a translucent overlay with `text` centered in the window.
     """
     if env.renderer is None or env.renderer.window is None:
         return
     surf = pygame.Surface(env.renderer.window.get_size(), pygame.SRCALPHA)
     surf.fill((0, 0, 0, 180))  # semi-transparent black
     txt  = FONT.render(text, True, (255, 255, 255))
     surf.blit(txt, txt.get_rect(center=surf.get_rect().center))
     env.renderer.window.blit(surf, (0, 0))
     pygame.display.flip()
 
 
 def _load_model_async():
     """Background thread target for loading the PPO checkpoint."""
     global loaded_model, load_exception
     try:
-        # Load the PPO checkpoint without attaching the live environment.
-        # stable_baselines3 will access the environment's spaces during load,
-        # which is safe, but calling env.reset() inside this background thread
-        # can freeze Pygame.  By loading without ``env`` and assigning it on
-        # the main thread, we avoid any Pygame calls outside the main loop.
+        # Load the PPO checkpoint without creating or touching any env.
+        # Accessing a live ``TetrisEnv`` from this background thread can
+        # freeze Pygame, so we simply load the weights and use them later on
+        # the main thread for inference.
         loaded_model = PPO.load(BEST_MODEL, device=device)
     except Exception as e:
         load_exception = e
 
 # ── Main loop ─────────────────────────────────────────────────────
 while True:
     # 1) Service Pygame events to keep the window responsive
     for ev in pygame.event.get():
         if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
             pygame.quit()
             sys.exit()
 
     # 2) Load checkpoints asynchronously so the window never freezes
     if load_thread is not None:
         if load_thread.is_alive():
             overlay("Loading best model …")
             time.sleep(0.1)
             continue
         load_thread.join()
         if load_exception:
             print("❌  Failed to load checkpoint:", load_exception)
             load_exception = None
             load_thread = None
             time.sleep(1.0)
             continue
         model = loaded_model
-        if model is not None:
-            # Attach the live environment on the main thread where all
-            # Pygame interactions happen. Calling set_env here ensures we
-            # avoid any Pygame calls from the loader thread.
-            model.set_env(env)
         loaded_model = None
         last_hash = hash_being_loaded
         print(f"🔄  Reloaded {BEST_MODEL}  (hash {last_hash[:8]})")
         load_thread = None
         continue
 
     if not BEST_MODEL.exists():
         overlay("Waiting for checkpoints/best_model.zip …")
         time.sleep(1.0)
         continue
 
     current_hash = md5(BEST_MODEL)
     if current_hash != last_hash:
         overlay("Loading best model …")
         hash_being_loaded = current_hash
         load_thread = threading.Thread(target=_load_model_async)
         load_thread.start()
         continue
 
     # 3) Play one episode, then loop back for the next
     try:
         obs, _ = env.reset()   # Always reset at the start of each new episode
         done = False
         while not done:
             # keep window responsive inside the episode
