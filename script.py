import subprocess
import json
import time
import requests
import csv
import os
import random
import platform
from collections import defaultdict

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
        "--jinja",
        "--chat-template-file", "phi4.jinja"  # Força o uso do template phi4.jinja
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
    
def sample_data(data, config):
    """Seleciona um subconjunto de dados de forma aleatória e balanceada por categoria."""
    if not config.get('enabled', False):
        return data

    questions_per_category = config.get('questions_per_category', 5)  # 5 questões por categoria por padrão
    
    print(f"Iniciando amostragem com {questions_per_category} questões por categoria...")
    print(f"Isso gerará {questions_per_category * 2} logs por categoria (2 logs por questão)...")

    # Agrupa os prompts por categoria
    categorized_data = defaultdict(list)
    for item in data:
        category = item.get('category', item.get('task', 'unknown'))
        categorized_data[category].append(item)

    sampled_data = []
    total_questions = 0
    total_logs = 0
    
    # Seleciona aleatoriamente um número fixo de questões de cada categoria
    for category, items in categorized_data.items():
        # Garante que a semente aleatória seja a mesma para reprodutibilidade
        random.seed(config.get('random_seed', 42))
        
        if len(items) > questions_per_category:
            sampled_items = random.sample(items, questions_per_category)
        else:
            # Pega todos se houver menos que o solicitado
            sampled_items = items
            print(f"AVISO: Categoria '{category}' tem apenas {len(items)} questões (solicitado: {questions_per_category})")
        
        sampled_data.extend(sampled_items)
        category_questions = len(sampled_items)
        category_logs = category_questions * 2
        total_questions += category_questions
        total_logs += category_logs
        
        print(f"  Categoria '{category}': {category_questions} questões = {category_logs} logs")

    # Embaralha a lista final para garantir que a ordem das categorias seja aleatória
    random.shuffle(sampled_data)

    print(f"Amostragem concluída:")
    print(f"  Total de questões selecionadas: {total_questions}")
    print(f"  Total de logs que serão gerados: {total_logs}")
    print(f"  Categorias encontradas: {len(categorized_data)}")
    
    return sampled_data

def main():
    with open("config.json", "r") as f:
        config = json.load(f)

    # Carrega todos os dados do MT-Bench
    all_mt_bench_data = load_mt_bench_data(config)
    if not all_mt_bench_data:
        return

    # Aplica a amostragem aos dados carregados
    mt_bench_data = sample_data(all_mt_bench_data, config.get('sampling_config', {}))

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
                    category = item.get('category', item.get('task', 'unknown'))
                    
                    if 'history' in item and isinstance(item['history'], list):
                        turns = [h['user'] for h in item['history'] if 'user' in h]
                    else:
                        turns = item.get('turns', [])

                    if len(turns) < 2:
                        continue

                    # Extrai as perguntas originais para ambos os turnos
                    raw_question1 = turns[0]
                    raw_question2 = turns[1]

                    # --- Turno 1 ---
                    # Formata o prompt para o primeiro turno
                    prompt1_formatted = f"<|im_start|>user\n{raw_question1}<|im_end|>\n<|im_start|>assistant\n"
                    print(f"\nTestando: ID={question_id}, Cat={category}, Turno=1")
                    
                    response1_data = run_inference(config, prompt1_formatted)
                    if not response1_data: continue

                    # Coleta de métricas do Turno 1
                    timing1 = response1_data.get('timings', {})
                    predicted_ms = timing1.get('predicted_ms', 0)
                    predicted_n = timing1.get('predicted_n', 0)
                    prompt_ms = timing1.get('prompt_ms', 0)
                    tps1 = timing1.get('predicted_per_second', 0)
                    if predicted_n == 1 and predicted_ms < 1.0:
                        tps1 = min(tps1, 100)
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

                    # Constrói o histórico da conversa para o segundo turno usando as perguntas originais
                    full_prompt_turn2 = (
                        f"<|im_start|>user\n{raw_question1}<|im_end|>\n"
                        f"<|im_start|>assistant\n{response1_content}<|im_end|>\n"
                        f"<|im_start|>user\n{raw_question2}<|im_end|>\n"
                        f"<|im_start|>assistant\n"
                    )
                    
                    print(f"Testando: ID={question_id}, Cat={category}, Turno=2")
                    response2_data = run_inference(config, full_prompt_turn2)
                    if not response2_data: continue
                    
                    # Coleta de métricas do Turno 2
                    timing2 = response2_data.get('timings', {})
                    predicted_ms2 = timing2.get('predicted_ms', 0)
                    predicted_n2 = timing2.get('predicted_n', 0)
                    prompt_ms2 = timing2.get('prompt_ms', 0)
                    tps2 = timing2.get('predicted_per_second', 0)
                    if predicted_n2 == 1 and predicted_ms2 < 1.0:
                        tps2 = min(tps2, 100)
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

    # --- Seção de Desligamento ---
    shutdown_config = config.get('shutdown_on_completion', {})
    if shutdown_config.get('enabled', False):
        delay = shutdown_config.get('delay_seconds', 15)
        print("-" * 50)
        print(f"AVISO: O computador será desligado em {delay} segundos.")
        print("Pressione Ctrl+C para cancelar.")
        print("-" * 50)
        
        try:
            for i in range(delay, 0, -1):
                print(f"Desligando em {i}...")
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nDesligamento cancelado pelo usuário.")
            return

        system_os = platform.system()
        shutdown_command = ""
        if system_os == "Windows":
            shutdown_command = "shutdown /s /t 1"
        elif system_os == "Linux" or system_os == "Darwin": # Darwin é o macOS
            shutdown_command = "sudo shutdown -h now"
        else:
            print(f"Sistema operacional '{system_os}' não suportado para desligamento automático.")
            return

        print(f"Executando comando de desligamento: {shutdown_command}")
        # Adicionar um aviso sobre a necessidade de permissões de administrador
        if system_os != "Windows":
            print("Nota: Este comando pode exigir privilégios de administrador (sudo).")
        
        os.system(shutdown_command)

if __name__ == "__main__":
    main()
