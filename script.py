import subprocess
import json
import time
import requests
import csv
import os

def load_mt_bench_data(config):
    """Baixa e carrega os dados do MT-Bench 101 se não existirem localmente."""
    local_file = config['local_data_file']
    if not os.path.exists(local_file):
        print(f"Baixando dataset de {config['mt_bench_url']}...")
        try:
            response = requests.get(config['mt_bench_url'])
            response.raise_for_status()
            with open(local_file, 'w', encoding='utf-8') as f:
                f.write(response.text)
            print("Download concluído.")
        except requests.RequestException as e:
            print(f"Erro ao baixar o dataset: {e}")
            return None
            
    # Carrega os dados do arquivo .jsonl
    data = []
    with open(local_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"Carregados {len(data)} prompts do MT-Bench 101.")
    return data

def start_server(config, model_path, threads):
    """Inicia o servidor llama.cpp como um processo em segundo plano."""
    server_url = f"http://{config['server_host']}:{config['server_port']}"
    command = [
        config['llama_server_path'],
        "-m", model_path,
        "-c", "4096", # Contexto maior para conversas
        "-t", str(threads),
        "--host", config['server_host'],
        "--port", str(config['server_port']),
        "--chat-template", "phi4"
    ]
    print(f"Iniciando servidor: {' '.join(command)}")
    server_process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    
    # Espera o servidor estar pronto
    max_retries = 45
    for i in range(max_retries):
        health_check_url = f"http://{config['server_host']}:{config['server_port']}/health"
        try:
            response = requests.get(health_check_url)
            if response.json().get("status") == "ok":
                print(f"Servidor iniciado com sucesso no PID: {server_process.pid}")
                return server_process
        except requests.RequestException:
            pass
        time.sleep(1)
        print(f"Aguardando o servidor... ({i+1}/{max_retries})")
    
    print("Erro: Servidor não iniciou a tempo.")
    # Imprime a saída do servidor para depuração
    output, _ = server_process.communicate()
    print(output.decode('utf-8'))
    server_process.terminate()
    return None

def stop_server(process):
    """Para o processo do servidor."""
    if process:
        print(f"Parando servidor com PID: {process.pid}")
        process.terminate()
        process.wait()
        print("Servidor parado.")

def run_inference(config, prompt):
    """Envia um prompt para o servidor e retorna a resposta completa."""
    server_url = f"http://{config['server_host']}:{config['server_port']}/completion"
    headers = {"Content-Type": "application/json"}
    payload = {
        "prompt": prompt,
        "n_predict": config['tokens_to_generate'],
        "temperature": 0.2, # Temperatura baixa para respostas mais previsíveis
        "stop": ["<|im_end|>", "</s>", "User:"], # Tokens de parada para chat
        "stream": False
    }
    
    print(f"DEBUG: Enviando prompt (primeiros 100 chars): {prompt[:100]}...")
    
    try:
        response = requests.post(server_url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        response_data = response.json()
        
        # Mostrar payload de retorno ordenado
        print("=" * 80)
        print("PAYLOAD DE RETORNO ORDENADO:")
        print("=" * 80)
        for key in sorted(response_data.keys()):
            value = response_data[key]
            if isinstance(value, dict):
                print(f"{key}:")
                for subkey in sorted(value.keys()):
                    print(f"  {subkey}: {value[subkey]}")
            elif isinstance(value, list):
                print(f"{key}: {value[:3]}..." if len(value) > 3 else f"{key}: {value}")
            else:
                print(f"{key}: {value}")
        print("=" * 80)
        
        return response_data
    except requests.RequestException as e:
        print(f"Erro ao fazer requisição: {e}")
        return None

def main():
    with open("config.json", "r") as f:
        config = json.load(f)

    mt_bench_data = load_mt_bench_data(config)
    if not mt_bench_data:
        return

    with open(config['output_csv'], 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'model_name', 'threads', 'question_id', 'category', 'turn',
            'ttft_ms', 'tpot_ms', 'tokens_per_sec', 'prompt_tokens', 'generated_tokens'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for model_config in config['models']:
            threads = config['threads']  # Use thread value from config
            server_process = start_server(config, model_config['path'], threads)
            if not server_process:
                continue

            for item in mt_bench_data:
                    question_id = item['id']
                    # Determine category, defaulting to task if category missing
                    category = item.get('category', item.get('task', 'unknown'))
                    # Build turn prompts from history if available, else use legacy 'turns'
                    if 'history' in item and isinstance(item['history'], list):
                        turns = [h['user'] for h in item['history'] if 'user' in h]
                    else:
                        turns = item.get('turns', [])
                    # Need at least two user prompts for two turns
                    if len(turns) < 2:
                        continue
                    # --- Turno 1 ---
                    # Use extracted prompts
                    prompt1 = turns[0]
                    print(f"\nTestando: ID={question_id}, Cat={category}, Turno=1")
                    
                    response1_data = run_inference(config, prompt1)
                    if not response1_data: continue

                    # Coleta de métricas do Turno 1
                    timing1 = response1_data.get('timings', {})
                    predicted_ms = timing1.get('predicted_ms', 0)
                    predicted_n = timing1.get('predicted_n', 0)
                    prompt_ms = timing1.get('prompt_ms', 0)
                    
                    # Usar o valor já calculado pelo servidor quando disponível
                    tps1 = timing1.get('predicted_per_second', 0)
                    
                    # Para casos especiais (1 token, EOS), usar uma abordagem mais conservadora
                    if predicted_n == 1 and predicted_ms < 1.0:
                        # Muito provável que seja um EOS token imediato - usar valor mais baixo
                        tps1 = min(tps1, 100)  # Limitar a 100 T/s no máximo
                    
                    tpot1 = predicted_ms / predicted_n if predicted_n > 0 else 0
                    ttft1 = prompt_ms
                    
                    writer.writerow({
                        'model_name': model_config['name'], 'threads': threads,
                        'question_id': question_id, 'category': category, 'turn': 1,
                        'ttft_ms': ttft1, 'tpot_ms': tpot1, 'tokens_per_sec': tps1,
                        'prompt_tokens': timing1.get('prompt_n', 0),
                        'generated_tokens': timing1.get('predicted_n', 0)
                    })
                    print(f"Turno 1: {tps1:.2f} T/s | Gerados: {predicted_n} tokens | Tempo: {predicted_ms:.2f}ms")
                    
                    # --- Turno 2 ---
                    response1_content = response1_data.get('content', '')
                    prompt2_question = turns[1]

                    # Constrói o histórico da conversa para o segundo turno
                    # O formato depende do --chat-template usado no servidor
                    # Para 'chatml', o formato é o seguinte:
                    full_prompt_turn2 = f"<|im_start|>user\n{prompt1}<|im_end|>\n<|im_start|>assistant\n{response1_content}<|im_end|>\n<|im_start|>user\n{prompt2_question}<|im_end|>\n<|im_start|>assistant\n"
                    
                    print(f"Testando: ID={question_id}, Cat={category}, Turno=2")
                    response2_data = run_inference(config, full_prompt_turn2)
                    if not response2_data: continue
                    
                    # Coleta de métricas do Turno 2
                    timing2 = response2_data.get('timings', {})
                    predicted_ms2 = timing2.get('predicted_ms', 0)
                    predicted_n2 = timing2.get('predicted_n', 0)
                    prompt_ms2 = timing2.get('prompt_ms', 0)
                    
                    # Usar o valor já calculado pelo servidor quando disponível
                    tps2 = timing2.get('predicted_per_second', 0)
                    
                    # Para casos especiais (1 token, EOS), usar uma abordagem mais conservadora
                    if predicted_n2 == 1 and predicted_ms2 < 1.0:
                        # Muito provável que seja um EOS token imediato - usar valor mais baixo
                        tps2 = min(tps2, 100)  # Limitar a 100 T/s no máximo
                    
                    tpot2 = predicted_ms2 / predicted_n2 if predicted_n2 > 0 else 0
                    ttft2 = prompt_ms2
                    
                    writer.writerow({
                        'model_name': model_config['name'], 'threads': threads,
                        'question_id': question_id, 'category': category, 'turn': 2,
                        'ttft_ms': ttft2, 'tpot_ms': tpot2, 'tokens_per_sec': tps2,
                        'prompt_tokens': timing2.get('prompt_n', 0),
                        'generated_tokens': timing2.get('predicted_n', 0)
                    })
                    print(f"Turno 2: {tps2:.2f} T/s | Gerados: {predicted_n2} tokens | Tempo: {predicted_ms2:.2f}ms")

            stop_server(server_process)
            print("-" * 50)

    print(f"Benchmark MT-Bench 101 concluído! Resultados salvos em {config['output_csv']}")

if __name__ == "__main__":
    main()