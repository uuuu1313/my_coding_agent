from dotenv import load_dotenv
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from web_code_agent_graph import graph

load_dotenv()

st.title("깃허브 통합 검색 에이전트")
st.markdown("#### Intelligent Reasearch Assistant!")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        AIMessage(content="안녕하세요! 깃허브 통합 검색 에이전트입니다. 질문을 입력해주세요.")
    ]

# 메세지 히스토리 표시
for msg in st.session_state.messages:
    if isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)

# 사용자 입력 처리
if prompt := st.chat_input("질문을 입력해주세요"):
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.chat_message("user").write(prompt)

    # AI 응답처리
    with st.chat_message("assistant"):
        initial_state = {
            "question": prompt,
            "certainty_score": 0,
            "search_results": [],
            "web_score": "",
            "repo_name": "",
            "generation": ""
        }

        try:
            # 그래프 실행 및 상태 업데이트
            for step in graph.stream(
                initial_state,
                config={
                    "recursion_limit": 100
                }
            ):
                # 현재 단계 표시
                for node_name, state in step.items():
                    # 확실성 점수 표시
                    if "certainty_score" in state:
                        with st.status("제가 스스로 답할 수 있는지 고민중이에요...", expanded=True) as status:
                            st.write(f"LLM의 확신 정도 : {state['certainty_score']}/100")
                            if state['certainty_score'] == 100:
                                status.update(label="이건 확실히 알겠네요! 답변 해볼게요!", state="complete", expanded=False)
                            else:
                                status.update(label="이건 제가 잘모르는거네요... 웹 검색을 해볼게요!", state="complete", expanded=False)
                    
                    # 웹 검색 결과표시
                    if "web_score" in state:
                        if state["web_score"] == "yes":
                            with st.status("웹에서 한번 검색해볼게요...", expanded=True) as status:
                                status.update(label="오! 웹 검색 결과 유용한 정보가 있었어요!", state="complete", expanded=False)
                            with st.expander("웹 검색 결과 : "):
                                for i, result in enumerate(state["search_results"], 1):
                                    st.write(f"Source {i}:")
                                    st.write(f"URL : {result['url']}")
                        else:
                            with st.status("웹 검색으로 해결한지 확인중...", expanded=False) as status:
                                status.update(label="흠.. 웹 검색만으로는 어려워요. 깃헙 레포지토리를 찾아볼게요!")
                    
                    # GitHub 저장소 정보 표시
                    if 'repo_name' in state:
                        with st.expander("깃헙 검색 결과"):
                            st.write(f"참고한 Github 저장소 : {state['repo_name']}")
                    
                    # 최종 생성된 답변 처리
                    if 'generation' in state:
                        last_msg = state['generation']
                        st.session_state.messages.append(AIMessage(content=last_msg))
                        st.markdown(last_msg)
        except Exception as e:
            st.error(f"오류 발생 : {e}")


            
                            
