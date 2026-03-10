import os
import json
from dotenv import load_dotenv

# 匯入您的三個核心模組
from crew import CompetitiveAnalysisCrew
from langGraph import build_marketing_graph
# 修正：匯入正確的類別名稱 ProductVisualCrew (原本檔名為 crew_image.py)
from crew_image import ProductVisualCrew

# 載入環境變數
load_dotenv()

def run():
    # 設定初始參數
    product_input = "Penhaligon's 潘海利根獸首肖像香水系列 Lord George 公鹿淡香精 75ml"
    source_image = "./800x.webp"  # 您的原始照片路徑
    output_json = "marketing_result.json"

    print("🚀 [Step 1] 啟動 CrewAI：深入分析競品中...")
    try:
        # 1. 執行 CrewAI 分析 (市場戰略部分)
        crew_instance = CompetitiveAnalysisCrew().crew()
        crew_output = crew_instance.kickoff(inputs={"product_description": product_input})
        strategy_report = str(crew_output.raw)
        print(f"✅ CrewAI 戰略報告產出完成 (長度：{len(strategy_report)} 字)")

        print("\n🎨 [Step 2] 啟動 LangGraph：生成整合行銷文案...")
        # 2. 建立 LangGraph 並執行 (文案創意部分)
        marketing_app = build_marketing_graph()
        initial_state = {
            "strategy_report": strategy_report,
            "retry_count": 0
        }
        
        final_result_data = None
        for event in marketing_app.stream(initial_state, config={"recursion_limit": 20}):
            for node_name, output in event.items():
                print(f"   > 執行節點: {node_name}")
                if node_name == "output":
                    final_result_data = output

        # 3. 檢查並提取文案結果
        if final_result_data and "final_options" in final_result_data:
            options = final_result_data["final_options"]
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(final_result_data, f, ensure_ascii=False, indent=4)
            
            # 取得第一個最優文案選項
            best_option = options[0]
            product_name = best_option.get("product_name", product_input)
            hook_text = best_option.get("hook_line", "")

            print(f"✅ 文案已儲存至 {output_json}")
            print(f"🎯 提取 Hook: {hook_text}")

            print("\n📸 [Step 3] 啟動 CrewAI Visual Team：生成商業級產品圖...")
            # 4. 執行圖片生成團隊邏輯
            # 修正：準備符合 ProductVisualCrew 期待的 inputs 字典
            visual_inputs = {
                "product_name": product_name,
                "copywriting": hook_text,
                "source_image_path": source_image
            }

            # 修正：實例化 ProductVisualCrew 並執行 kickoff
            visual_crew_instance = ProductVisualCrew().crew()
            visual_crew_instance.kickoff(inputs=visual_inputs)

            print("\n" + "★" * 50)
            print("✨ 全自動行銷整合流程結束 ✨")
            print(f"📦 最終文案：{output_json}")
            print(f"🖼️ 最終圖片：已根據 Crew 任務設定儲存")
            print("★" * 50)

        else:
            print("⚠️ LangGraph 流程未獲得有效結果，停止圖片生成。")

    except Exception as e:
        print(f"❌ 流程發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run()