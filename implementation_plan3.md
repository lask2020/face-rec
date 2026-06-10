# Golang Migration Plan (600-Camera Scale)

การย้ายระบบไปใช้ **Golang** สำหรับรองรับกล้องจำนวนมาก (600+ ตัว) เป็นการตัดสินใจที่ถูกต้องที่สุดครับ เนื่องจาก Golang มีจุดเด่นด้าน Concurrency (`Goroutines`) ที่สามารถรับโหลด RTSP Stream พร้อมกันเป็นพันๆ สตรีมได้โดยกิน RAM และ CPU น้อยกว่า Python อย่างมหาศาล

อย่างไรก็ตาม **งานด้าน AI (InsightFace)** ยังคงเหมาะสมที่จะอยู่ใน Python เนื่องจาก Library ที่สมบูรณ์แบบกว่า ดังนั้นสถาปัตยกรรมที่ดีที่สุดระดับ Enterprise คือการทำ **Hybrid Microservices**

## Proposed Architecture (Fully Distributed)

1. **Golang Control Plane (API Gateway & Manager)**
   - จัดการ State ของระบบ, เก็บข้อมูลลง Database (PostgreSQL)
   - จัดการ WebSocket และให้บริการ RESTful API แก่หน้าจอ Frontend
   - ทำงานเหมือน "หัวหน้างาน" คอยจ่ายคิวและแบ่งโหลดกล้อง 600 ตัวให้เหล่า Worker Nodes รับไปทำ
   
2. **Golang Ingestion Worker Nodes (Fleet of Stream Grabbers)**
   - แยกเครื่องเซิร์ฟเวอร์ย่อยออกมาทำหน้านี้โดยเฉพาะ (เช่น 1 Node รับผิดชอบ 50-100 กล้อง)
   - ดึง RTSP Streams ตามคำสั่งที่ได้รับจาก Control Plane
   - ถอดรหัสวิดีโอ ดึงรูปภาพ 1-2 FPS แล้วโยนใส่ Message Queue (เช่น Kafka, RabbitMQ หรือ Redis Streams)

3. **Python AI Inference Nodes (GPU/CPU AI Fleet)**
   - รอรับภาพจาก Message Queue ที่ Ingestion Nodes โยนเข้ามา
   - ทำ Face Detection + Recognition ด้วย `InsightFace` โดยสามารถรับภาพแบบข้ามกล้องมาทำ Batching ทีเดียวเพื่อให้ประสิทธิผลของ GPU ออกมาสูงสุด
   - ส่งผลลัพธ์การจับคู่ใบหน้า (Face IDs) กลับไปให้ Golang Control Plane ผ่านทาง Message Queue หรือ gRPC

## User Review Required

> [!WARNING]  
> การเขียน AI ด้วยภาษา Go ล้วนๆ (Pure Go) ทำได้ยากมาก เพราะโมเดล InsightFace ต้องพึ่งพา `numpy` และการทำ Post-processing (คำนวณเวกเตอร์ 512 มิติ และ FAISS) ที่ซับซ้อนมากในภาษา Python 
> 
> **ดังนั้นแผนนี้จะมุ่งเน้นไปที่การสร้าง "Golang Service" ขึ้นมาเพื่อนำร่องจัดการเรื่อง RTSP Streams และ API แยกส่วนกันครับ**

## Proposed Changes

### Phase 1: จัดเตรียม Message Broker & Database
- เปลี่ยนจาก SQLite เป็น **PostgreSQL**
- ติดตั้ง **Redis** เพื่อใช้เป็นสื่อกลางการส่งภาพข้ามระหว่าง Go กับ Python (In-memory, ความหน่วงต่ำสุดๆ)

### Phase 2: พัฒนาส่วน Master/Control Plane (Golang)
- สร้างบริการจัดการ API และมอบหมายกล้อง
- ออกแบบชิ้นงาน (Task queue) สำหรับกระจายให้ฝูง Worker

### Phase 3: พัฒนา Golang Ingestion Worker Nodes [NEW]
สร้าง Worker Service ใน Go (Deploy แยกเป็น Container อิสระกี่ตัวก็ได้)
- เขียนระบบดึงภาพจาก RTSP `go2rtc` หรือ FFmpeg-binding
- แต่ละเครื่องจะรับ Assignment จาก Control Plane (เช่น เครื่อง 1 รับกล้อง 1-100)
- รีดภาพยัดลง Message Queue อย่างรวดเร็ว

### Phase 4: ปรับแต่ง Python Face Engine [MODIFY]
- ลบระบบดึงกล้องเดิม (`stream_processor.py`) ทิ้ง
- เปลี่ยนให้เป็น "AI Worker Mode" ที่คอยดึงภาพจาก Queue มาเข้าคิวรัน AI อย่างเดียว
- สเกลแยกออกมาเป็น Container ย่อยได้หลายๆ เครื่องตามจำนวน GPU ที่มี

### Phase 4: ย้าย Logic ส่วน API และ Frontend สลับไปคุยกับ Go
- เขียนเทียบ RESTful API (การจัดการกล้อง, ดึงประวัติ, ฐานข้อมูล) ทั้งหมดด้วย Go Fiber หรือ Gin
- ย้าย WebSocket ให้ Go เป็นคนส่ง

## Open Questions

> [!IMPORTANT]
> 1. คุณต้องการให้เริ่มสร้างโปรเจกต์โครงสร้าง Golang เบื้องต้น (Phase 2 & 3) ตอนนี้เลยหรือไม่?
> 2. เครื่องที่ใช้พัฒนาในปัจจุบันมีการลง **Go** (Golang compiler) เอาไว้เตรียมพร้อมหรือยังครับ?
> 3. หรือต้องการชิมลางด้วยการแปลงโค้ด FastAPI หลักๆ มารันบน Go ดูก่อนแบบเล็กๆ ครับ?

## Verification Plan

### Automated/Manual Verification
- [ ] ทดสอบสร้างระบบ Golang ดึงภาพ RTSP ลิสต์รายชื่อกล้องจาก Synology
- [ ] ทดสอบว่าภาพจาก Go ถูกส่งไปประเมินหน้าใน Python ได้ครบถ้วน
- [ ] สังเกตปริมาณ RAM และ CPU การใช้พลังงานที่ลดลงอย่างชัดเจนเมื่อเทียบกับ Python threading ล้วน
