from collections import namedtuple
import pyomo.environ as pyomo
from pyomo.opt import SolverStatus, TerminationCondition
import pandas
import os
import math
from timeit import default_timer as timer
import argparse
import sys
import json

def executar_heuristic_sequencial():
    tempo_global_inicio = timer()

    parser = argparse.ArgumentParser()
    parser.add_argument("nGat", type=int)
    parser.add_argument("nPla", type=int)
    parser.add_argument("nDev", type=int)
    parser.add_argument("seed", type=int)
    parser.add_argument("qos_bound", type=float)
    args = parser.parse_args()

    nGat = args.nGat
    nPla = args.nPla
    nDev_inicial = args.nDev
    seed = args.seed
    qos_bound = args.qos_bound
    
    TOTAL_DEVICES_INICIAL = nDev_inicial

    PACKET_SIZE = 400  # bits
    QOS_LOWER_BOUND = qos_bound
    MAX_RKC = 6835.94
    MIN_RKC = 183.11
    MAX_DELAY = PACKET_SIZE / MIN_RKC

    PESO_COLETA = 1000.0

    cwd = os.getcwd()
    path_output = cwd + "/data/model/output/"
    path_data = cwd + "/data/model/"
    gateway_data = cwd + "/data/placement/"
    devices_data = cwd + "/data/placement/"

    Position = namedtuple('Position', ['x', 'y', 'z'])
    Configuration = namedtuple('Configuration', ['sf', 'tp'])
    Gateway = namedtuple('Gateway', ['bandwidths', 'max_datarates'])

    if nPla == 1:
        gateway_position_file = gateway_data + f"equidistantPlacement_{nGat}.dat"
    else:
        gateway_position_file = gateway_data + f"equidistantPlacement_{nGat}x{nPla}.dat"

    if not os.path.exists(gateway_position_file):
        print(f"[ERRO CRÍTICO] Arquivo de gateways não encontrado: {gateway_position_file}")
        sys.exit(1)

    gateway_position_df = pandas.read_csv(gateway_position_file, names=['x', 'y', 'z'], sep=" ", index_col=False)
    gateway_positions = {
        row.Index : Position(x=row.x, y=row.y, z=row.z)
        for row in gateway_position_df.itertuples()
    }

    slice_association_file = path_data + f"skl_{seed}s_{nGat}x{nPla}Gv_{nDev_inicial}D.dat"
    if not os.path.exists(slice_association_file):
        print(f"[ERRO CRÍTICO] Arquivo de slices não encontrado: {slice_association_file}")
        sys.exit(1)
        
    slice_association_df = pandas.read_csv(slice_association_file, names=['device', 'slice'], sep=" ", index_col=False)
    slice_associations = {row.device: row.slice for row in slice_association_df.itertuples()}

    end_device_position_file = devices_data + f"endDevices_LNM_Placement_{seed}s+{nDev_inicial}d.dat"
    if not os.path.exists(end_device_position_file):
        print(f"[ERRO CRÍTICO] Arquivo de sensores não encontrado: {end_device_position_file}")
        sys.exit(1)
        
    end_device_position_df = pandas.read_csv(end_device_position_file, names=['x', 'y', 'z'], sep=" ", index_col=False)
    master_end_device_positions = {
        row.Index: Position(x=row.x, y=row.y, z=row.z)
        for row in end_device_position_df.itertuples()
    }

    sensores_ativos = list(master_end_device_positions.keys())
    print(f"[INICIALIZAÇÃO] {len(sensores_ativos)} sensores carregados do disco.")

    slices = {0: 'slice1', 1: 'slice2', 2: 'slice3'}
    gateways = {0: Gateway(bandwidths=[125000.0 for _ in slices], max_datarates=[15197.75390625 for _ in slices])}

    spreading_factors = range(7, 13)
    transmission_powers = range(2, 16, 2)
    configurations = {
        idx: Configuration(sf, tp) 
        for idx, (sf, tp) in enumerate((sf, tp) for sf in spreading_factors for tp in transmission_powers)
    }
    sensitivity = {7: -130.0, 8: -132.5, 9: -135.0, 10: -137.5, 11: -140.0, 12: -142.5}
    referencePrx = 10.0
    referenceDistance = 1.0
    attenuationExponent = 3.76

    passo = 1
    trajetoria_log = []
    MAX_PASSOS = len(master_end_device_positions) + 5

    # -------------------------------------------------------------
    # LAÇO ROLLING HORIZON
    # -------------------------------------------------------------
    while len(sensores_ativos) > 0 and passo <= MAX_PASSOS:
        print(f"\n==================================================")
        print(f"--- PASSO {passo} | Sensores Pendentes: {len(sensores_ativos)} ---")
        print(f"==================================================")

        end_device_positions = {dev_id: master_end_device_positions[dev_id] for dev_id in sensores_ativos}

        # 1. PRÉ-FILTRAGEM FOCADA EM SF <= 7 PARA O SOLVER
        valid_tuples = []
        for g in gateways.keys():
            for p, g_pos in gateway_positions.items():
                for k, d_pos in end_device_positions.items():
                    s_id = slice_associations.get(k, -1)
                    if s_id not in slices:
                        continue

                    dist = math.sqrt((g_pos.x - d_pos.x)**2 + (g_pos.y - d_pos.y)**2 + (g_pos.z - d_pos.z)**2) / referenceDistance
                    pl = 10.0 * attenuationExponent * math.log10(dist / referenceDistance) if dist > referenceDistance else 0.0

                    for c, cfg in configurations.items():
                        if cfg.sf > 8:
                            continue

                        prx_est = cfg.tp - referencePrx - pl
                        if prx_est < sensitivity[cfg.sf]:
                            continue

                        rkc = cfg.sf * gateways[g].bandwidths[s_id] / (2 ** cfg.sf)
                        delay = PACKET_SIZE / rkc
                        qos_val = (4.0 / 5.0 * rkc / MAX_RKC) + (1.0 - (delay / MAX_DELAY))
                        if qos_val < QOS_LOWER_BOUND:
                            continue

                        valid_tuples.append((g, p, k, c))

        if not valid_tuples:
            print("[ALERTA] Nenhum sensor pendente tem enlace viável com SF <= 7. Encerrando.")
            break

        # Verificação e isolamento de nós órfãos para SF7
        sensores_com_opcao = {t[2] for t in valid_tuples}
        orfaos = [k for k in sensores_ativos if k not in sensores_com_opcao]
        if orfaos:
            print(f"[ALERTA] {len(orfaos)} nós sem cobertura SF<=7 ignorados nesta rodada: {orfaos}")
            sensores_ativos = [d for d in sensores_ativos if d not in orfaos]
            if not sensores_ativos:
                break
            continue

        model = pyomo.ConcreteModel()
        model.VALID = pyomo.Set(initialize=valid_tuples, dimen=4)
        model.P = pyomo.Set(initialize=gateway_positions.keys())
        model.K = pyomo.Set(initialize=sensores_ativos)

        model.x = pyomo.Var(model.VALID, domain=pyomo.Binary)
        model.pos_ativa = pyomo.Var(model.P, domain=pyomo.Binary)
        model.atendido = pyomo.Var(model.K, domain=pyomo.Binary)

        # Restrição: exactamente 1 paragem ativa
        model.uma_parada = pyomo.Constraint(expr=sum(model.pos_ativa[p] for p in model.P) == 1)

        # Restrição: sensor só transmite se a posição estiver ativa
        model.link_pos = pyomo.ConstraintList()
        for (g, p, k, c) in model.VALID:
            model.link_pos.add(model.x[g, p, k, c] <= model.pos_ativa[p])

        # Restrição: atendimento no máximo uma vez
        model.atendimento = pyomo.ConstraintList()
        for k in model.K:
            x_k = [model.x[g, p, dev, c] for (g, p, dev, c) in model.VALID if dev == k]
            model.atendimento.add(sum(x_k) == model.atendido[k])

        # Restrição de capacidade por slice
        model.capacidade = pyomo.ConstraintList()
        for p in model.P:
            for s_id in slices.keys():
                termos = []
                g_terms = []
                for (g, pos, dev, c) in model.VALID:
                    if pos == p and slice_associations.get(dev, -1) == s_id:
                        sf = configurations[c].sf
                        taxa = sf * gateways[g].bandwidths[s_id] / (2 ** sf)
                        termos.append(model.x[g, pos, dev, c] * taxa)
                        g_terms.append(g)
                if termos:
                    g_alvo = g_terms[0]
                    model.capacidade.add(sum(termos) <= gateways[g_alvo].max_datarates[s_id])

        model.OBJ = pyomo.Objective(expr=- PESO_COLETA * sum(model.atendido[k] for k in model.K), sense=pyomo.minimize)

        opt = pyomo.SolverFactory('scip')
        result = opt.solve(model, options={'limits/time': 60, 'limits/gap': 0.01}, tee=False)

        if (result.solver.status != SolverStatus.ok) or \
           (result.solver.termination_condition not in (TerminationCondition.optimal, TerminationCondition.feasible)):
            print(f"[ERRO] Falha de convergência no passo {passo}. Abortando.")
            break

        pos_escolhida_id = None
        for p in model.P:
            if pyomo.value(model.pos_ativa[p]) and pyomo.value(model.pos_ativa[p]) > 0.5:
                pos_escolhida_id = p
                break

        if pos_escolhida_id is None:
            print("[ERRO] Nenhuma posição ativada. Encerrando.")
            break

        waypoint_pos = gateway_positions[pos_escolhida_id]

        # Nós recolhidos com SF <= 7
        recolhidos_sf7 = [
            k for (g, p, k, c) in model.VALID
            if p == pos_escolhida_id and pyomo.value(model.x[g, p, k, c]) > 0.5
        ]

        # -------------------------------------------------------------
        # AUDITORIA FÍSICA: AVALIAÇÃO DE TODOS OS NÓS COM VISADA (SF 7 a 12)
        # -------------------------------------------------------------
        sf_visada_todos = {}
        for k in sensores_ativos:
            d_pos = master_end_device_positions[k]
            s_id = slice_associations.get(k, -1)
            dist = math.sqrt((waypoint_pos.x - d_pos.x)**2 + (waypoint_pos.y - d_pos.y)**2 + (waypoint_pos.z - d_pos.z)**2) / referenceDistance
            pl = 10.0 * attenuationExponent * math.log10(dist / referenceDistance) if dist > referenceDistance else 0.0

            melhor_sf = None
            for c, cfg in configurations.items():
                prx_est = cfg.tp - referencePrx - pl
                if prx_est >= sensitivity[cfg.sf]:
                    if s_id in slices:
                        rkc = cfg.sf * gateways[0].bandwidths[s_id] / (2 ** cfg.sf)
                        delay = PACKET_SIZE / rkc
                        qos_val = (4.0 / 5.0 * rkc / MAX_RKC) + (1.0 - (delay / MAX_DELAY))
                        if qos_val >= QOS_LOWER_BOUND:
                            if melhor_sf is None or cfg.sf < melhor_sf:
                                melhor_sf = cfg.sf
            
            if melhor_sf is not None:
                sf_visada_todos[k] = melhor_sf

        nao_recolhidos = [k for k in sf_visada_todos.keys() if k not in recolhidos_sf7]

        print(f"[ROTA] Waypoint ID {pos_escolhida_id} -> ({waypoint_pos.x:.1f}, {waypoint_pos.y:.1f}, {waypoint_pos.z:.1f})")
        print(f"[RECOLHIDOS SF<=7] Total: {len(recolhidos_sf7)} -> {recolhidos_sf7}")
        print(f"[AUDITORIA VISADA] Nós no alcance: {list(sf_visada_todos.keys())}")
        print(f"[NÃO RECOLHIDOS (SF>7)] {nao_recolhidos}")

        trajetoria_log.append({
            "passo": passo,
            "gateway_id": pos_escolhida_id,
            "x": waypoint_pos.x,
            "y": waypoint_pos.y,
            "z": waypoint_pos.z,
            "qtd_recolhidos": len(recolhidos_sf7),
            "sensores_recolhidos_sf7": json.dumps(recolhidos_sf7),
            "sf_detalhado_visada": json.dumps(sf_visada_todos),
            "sensores_nao_recolhidos": json.dumps(nao_recolhidos)
        })

        if recolhidos_sf7:
            sensores_ativos = [d for d in sensores_ativos if d not in recolhidos_sf7]
        else:
            print("[ALERTA] Paragem sem recolha. Interrompendo para evitar ciclo infinito.")
            break

        passo += 1

    os.makedirs(path_output, exist_ok=True)
    df_traj = pandas.DataFrame(trajetoria_log)
    df_traj.to_csv(path_output + "trajetoria_uav_final.csv", index=False)

    tempo_global_fim = timer()
    print(f"\n==================================================")
    if len(sensores_ativos) == 0:
        print(f"[FINALIZADO] Missão concluída com 100% de sucesso!")
    else:
        print(f"[ENCERRADO] Missão finalizada. Restaram {len(sensores_ativos)} nós pendentes.")
    print(f"[TEMPO TOTAL] Duração da otimização: {(tempo_global_fim - tempo_global_inicio):.2f} segundos.")
    print(f"==================================================")

if __name__ == "__main__":
    executar_heuristic_sequencial()