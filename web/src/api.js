import axios from "axios";

const instance = axios.create({
    baseURL: 'http://127.0.0.1:8000',
    headers: { 'Content-Type': 'application/json' },
    timeout: 20000
});

export const list_node_tasks = async () => {
    return instance.get('/node-tasks');
};
