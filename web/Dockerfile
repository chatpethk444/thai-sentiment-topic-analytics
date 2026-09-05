# Node image for the Next.js dashboard (dev mode).
FROM node:20-slim
WORKDIR /web
COPY package.json ./
RUN npm install
COPY . .
EXPOSE 3000
ENV NEXT_PUBLIC_API_URL=http://api:8000
CMD ["npm", "run", "dev", "--", "-p", "3000", "-H", "0.0.0.0"]
